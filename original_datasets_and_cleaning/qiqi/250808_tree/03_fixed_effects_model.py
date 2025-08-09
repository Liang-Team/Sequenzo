# 序数回归模型：按国家分别回归
# 因变量: seniority (序数变量：Junior < Assistant < Regular < Senior < Leader < Chief or founder)
# 解释变量: rarity_score
# 控制变量: 除 worker_id, country, seniority, rarity_score 外的所有列（包含数值型控制，类别型控制做独热编码）
# 注意：seniority是序数变量，应使用序数回归而非OLS

import pandas as pd
import numpy as np
import os
import argparse

# 解析参数并设置默认 CSV 路径（脚本同目录）
script_dir = os.path.dirname(__file__)
default_csv = os.path.join(script_dir, "final_df.csv")

parser = argparse.ArgumentParser()
parser.add_argument("--final-csv", dest="final_csv", default=default_csv, help="Path to final_df.csv")
args = parser.parse_args()

final_df = pd.read_csv(args.final_csv)

try:
    from linearmodels.panel import PanelOLS
    HAS_LM = True
except Exception:
    HAS_LM = False

# 尝试导入序数回归库
try:
    from statsmodels.miscmodels.ordinal_model import OrderedModel
    HAS_ORDERED = True
except ImportError:
    HAS_ORDERED = False

try:
    import mord  # 另一个序数回归库
    HAS_MORD = True
except ImportError:
    HAS_MORD = False

import statsmodels.api as sm
import patsy

assert 'final_df' in globals(), "请确保已有 final_df 变量（包含 worker_id, country, year, seniority, rarity_score 等列）"

# 确保必要列存在
required_cols = {'worker_id','country','year','seniority','rarity_score'}
missing_req = required_cols - set(final_df.columns)
assert not missing_req, f"final_df 缺少必要列: {missing_req}"

# 预处理：去除极端缺失
work = final_df.copy()
# 只保留必要列与潜在控制列的有效行
essential = ['worker_id','country','year','seniority','rarity_score']
work = work.dropna(subset=['worker_id','year','seniority','rarity_score'])

# 将 seniority 编码为序数
seniority_order = ['Junior', 'Assistant', 'Regular', 'Senior', 'Leader', 'Chief or founder']
seniority_mapping = {level: i for i, level in enumerate(seniority_order)}

print("序数变量 seniority 的编码映射:")
for level, code in seniority_mapping.items():
    print(f"  {level} -> {code}")

# 检查是否有未知的 seniority 值
unknown_seniority = set(work['seniority'].dropna().unique()) - set(seniority_order)
if unknown_seniority:
    print(f"⚠️ 发现未知的 seniority 值: {unknown_seniority}")

# 创建序数编码的 seniority
work['seniority_ordinal'] = work['seniority'].map(seniority_mapping)
work = work.dropna(subset=['seniority_ordinal'])  # 去除无法编码的行

# 预处理后，加入性别二元编码（若存在 gender 列）
if 'gender' in work.columns:
    # 统一大小写/空白
    work['gender'] = work['gender'].astype(str).str.strip().str.lower()
    # female=1, male=0，其他值设为NaN
    work['gender_female'] = work['gender'].map({'female': 1.0, 'male': 0.0})
else:
    work['gender_female'] = np.nan  # 若没有也不报错

# 控制变量集合（排除主键与因变量/主解释变量）
exclude = set(['worker_id','country','year','seniority','rarity_score', 'gender'])
control_cols_all = [c for c in work.columns if c not in exclude]

# 分类变量（object 或 category）与数值变量
cat_controls = [c for c in control_cols_all if str(work[c].dtype) in ['object','category']]
num_controls = [c for c in control_cols_all if c not in cat_controls]

countries = list(pd.unique(work['country']))

results = []
models = {}

for ctry in countries:
    dfc = work[work['country'] == ctry].copy()
    if dfc.empty:
        continue

    # 将 worker_id 设为字符串，year 设为 int
    dfc['worker_id'] = dfc['worker_id'].astype(str).str.strip()
    dfc['year'] = dfc['year'].astype(int)

    # 构建设计矩阵：rarity_score + controls（类别控制独热编码）
    X_parts = [dfc[['rarity_score']]]

    # 数值控制
    if num_controls:
        X_parts.append(dfc[num_controls])

    # 类别控制做独热编码（drop_first 避免完全共线）
    if cat_controls:
        dummies = pd.get_dummies(dfc[cat_controls], drop_first=True, dtype=float)
        if not dummies.empty:
            X_parts.append(dummies)

    X = pd.concat(X_parts, axis=1)
    y_ordinal = dfc['seniority_ordinal']  # 使用序数编码的 seniority
    y_original = dfc['seniority']  # 保留原始文本用于参考

    # 去除任何仍有缺失的行
    valid = X.notna().all(axis=1) & y_ordinal.notna()
    X = X.loc[valid]
    y_ordinal = y_ordinal.loc[valid]
    y_original = y_original.loc[valid]
    dfc_valid = dfc.loc[valid]

    # 方法1: 尝试序数回归（Ordered Logit）
    ordered_success = False
    
    if HAS_ORDERED:
        try:
            # 确保所有数据都是数值型
            X_clean = X.copy()
            
            # 检查并转换所有列为数值型
            for col in X_clean.columns:
                if X_clean[col].dtype == 'object':
                    # 尝试转换为数值型
                    X_clean[col] = pd.to_numeric(X_clean[col], errors='coerce')
                elif X_clean[col].dtype == 'bool':
                    # 布尔类型转换为浮点数
                    X_clean[col] = X_clean[col].astype(float)
            
            # 去除转换后产生的NaN行
            mask = X_clean.notna().all(axis=1) & y_ordinal.notna()
            X_clean = X_clean.loc[mask]
            y_clean = y_ordinal.loc[mask].astype(int)
            
            print(f"  {ctry}: 数据清理后剩余 {len(y_clean)} 观测值")
            print(f"  X矩阵形状: {X_clean.shape}, 列类型: {dict(X_clean.dtypes)}")
            
            # 确保所有变量都是数值型（float或int）
            for col in X_clean.columns:
                if X_clean[col].dtype.kind not in 'biufc':
                    print(f"  警告：列 {col} 类型为 {X_clean[col].dtype}")
                    X_clean[col] = X_clean[col].astype(float)
            
            # OrderedModel 不需要手动添加常数项
            X_final = X_clean
            
            print(f"  准备运行序数回归，因变量范围: {y_clean.min()}-{y_clean.max()}")
            
            # 使用 statsmodels 的序数回归
            mod = OrderedModel(y_clean, X_final, distr='logit')
            res = mod.fit(disp=False)
            models[(ctry,'ordered_logit')] = res

            coef = res.params.get('rarity_score', np.nan)
            se = res.bse.get('rarity_score', np.nan)
            pval = res.pvalues.get('rarity_score', np.nan)
            pseudo_r2 = getattr(res, 'prsquared', np.nan)
            n = int(res.nobs)
            results.append({'country': ctry, 'method': 'OrderedLogit', 'coef_rarity': coef, 'se': se, 'pval': pval, 'PseudoR2': pseudo_r2, 'N': n})
            ordered_success = True
            print(f"✓ {ctry}: 序数回归（Ordered Logit）成功运行，系数={coef:.4f}, p值={pval:.4f}")
            
        except Exception as e:
            print(f"⚠ {ctry}: 序数回归失败: {str(e)}")
            print(f"  数据诊断：X形状={X.shape}, y唯一值={sorted(y_ordinal.dropna().unique())}")
            ordered_success = False
    
    # 方法2: 如果statsmodels的序数回归失败，尝试替代方法
    if not ordered_success:
        print(f"  {ctry}: 尝试替代的序数回归方法...")
        
        # 尝试简化模型：只使用核心变量
        try:
            # 只使用 rarity_score 和几个主要控制变量
            essential_controls = []
            if 'work_years' in num_controls:
                essential_controls.append('work_years')
            if 'gender_female' in num_controls:
                essential_controls.append('gender_female')
            
            X_simple = dfc[['rarity_score'] + essential_controls].copy()
            
            # 确保数据类型正确
            for col in X_simple.columns:
                if X_simple[col].dtype == 'object':
                    X_simple[col] = pd.to_numeric(X_simple[col], errors='coerce')
                elif X_simple[col].dtype == 'bool':
                    X_simple[col] = X_simple[col].astype(float)
            
            # 清理数据
            mask = X_simple.notna().all(axis=1) & y_ordinal.notna()
            X_simple_clean = X_simple.loc[mask]
            y_simple_clean = y_ordinal.loc[mask].astype(int)
            
            # 确保所有数据类型都是数值型
            for col in X_simple_clean.columns:
                if X_simple_clean[col].dtype.kind not in 'biufc':
                    X_simple_clean[col] = X_simple_clean[col].astype(float)
            
            print(f"  {ctry}: 简化模型，变量: {list(X_simple_clean.columns)}, 观测值: {len(y_simple_clean)}")
            
            # 序数回归（不添加常数项）
            mod_simple = OrderedModel(y_simple_clean, X_simple_clean, distr='logit')
            res_simple = mod_simple.fit(disp=False)
            models[(ctry,'ordered_logit_simple')] = res_simple

            coef = res_simple.params.get('rarity_score', np.nan)
            se = res_simple.bse.get('rarity_score', np.nan)
            pval = res_simple.pvalues.get('rarity_score', np.nan)
            pseudo_r2 = getattr(res_simple, 'prsquared', np.nan)
            n = int(res_simple.nobs)
            results.append({'country': ctry, 'method': 'OrderedLogit_Simple', 'coef_rarity': coef, 'se': se, 'pval': pval, 'PseudoR2': pseudo_r2, 'N': n})
            print(f"✓ {ctry}: 简化序数回归成功，系数={coef:.4f}, p值={pval:.4f}")
            
        except Exception as e2:
            print(f"✗ {ctry}: 所有序数回归方法都失败了: {str(e2)}")
            # 记录失败，但不做OLS回退
            results.append({'country': ctry, 'method': 'FAILED', 'coef_rarity': np.nan, 'se': np.nan, 'pval': np.nan, 'PseudoR2': np.nan, 'N': 0})
    
    print()  # 添加空行分隔不同国家的输出

fe_results_summary = pd.DataFrame(results).sort_values(['country']).reset_index(drop=True)
print('完成。各国 rarity_score 系数汇总:')
print(fe_results_summary)

# === 将结果写入一个 txt 报告 ===
import os, datetime

report_path = os.path.join(os.path.dirname(__file__), "fe_full_report.txt")
ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

with open(report_path, "w", encoding="utf-8") as f:
    f.write(f"Ordinal Regression Report\nGenerated: {ts}\n\n")
    f.write("IMPORTANT: seniority is an ordinal variable (Junior < Assistant < Regular < Senior < Leader < Chief or founder)\n\n")
    f.write("Statistical Methods Used:\n")
    f.write("1. Primary: Ordered Logit regression (if available)\n")
    f.write("2. Fallback: OLS with ordinally-encoded seniority + fixed effects\n\n")
    f.write("Controls used:\n")
    f.write(f"- Numeric controls: {num_controls}\n")
    f.write(f"- Categorical controls (one-hot): {cat_controls}\n\n")
    f.write("Seniority encoding:\n")
    for level, code in seniority_mapping.items():
        f.write(f"- {level} -> {code}\n")
    f.write("\nNotes:\n")
    f.write("- gender encoded as gender_female: female=1, male=0; other/unknown as NaN\n")
    f.write("- Ordered regression is more appropriate for ordinal outcomes than OLS\n")
    f.write("- Coefficients in ordered models represent log-odds of being in higher category\n\n")

    f.write("Country-level summary (one row per country):\n")
    f.write(fe_results_summary.to_string(index=False))
    f.write("\n\n")

    for ctry in countries:
        # 寻找该国家可用的模型
        key = None
        if (ctry, 'ordered_logit') in models:
            key = (ctry, 'ordered_logit')
        elif (ctry, 'ordered_logit_simple') in models:
            key = (ctry, 'ordered_logit_simple')
        
        if key is None:
            f.write(f"Country: {ctry} - 序数回归失败，无可用模型\n\n")
            continue
            
        f.write("=" * 80 + "\n")
        f.write(f"Country: {ctry} | Model: {key[1]}\n")
        f.write("=" * 80 + "\n")
        f.write(str(models[key].summary()))  # 调用summary()方法
        f.write("\n\n")

print(f"Full text report written to: {report_path}")