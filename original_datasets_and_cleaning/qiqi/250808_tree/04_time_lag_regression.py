# 时间滞后序数回归模型：t年的rarity_score预测t+1年的seniority
# 因变量: t+1年的seniority (序数变量：Junior < Assistant < Regular < Senior < Leader < Chief or founder)
# 解释变量: t年的rarity_score
# 控制变量: t年的其他变量
# 注意：由于需要下一年的seniority，最后一年的观测值会丢失

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

# 尝试导入序数回归库
try:
    from statsmodels.miscmodels.ordinal_model import OrderedModel
    HAS_ORDERED = True
except ImportError:
    HAS_ORDERED = False
    print("⚠️ statsmodels.miscmodels.ordinal_model 不可用，将无法运行序数回归")

import statsmodels.api as sm
import patsy

assert 'final_df' in globals(), "请确保已有 final_df 变量（包含 worker_id, country, year, seniority, rarity_score 等列）"

# 确保必要列存在
required_cols = {'worker_id','country','year','seniority','rarity_score'}
missing_req = required_cols - set(final_df.columns)
assert not missing_req, f"final_df 缺少必要列: {missing_req}"

print("创建时间滞后数据集...")
print(f"原始数据: {len(final_df)} 行")

# 将 seniority 编码为序数
seniority_order = ['Junior', 'Assistant', 'Regular', 'Senior', 'Leader', 'Chief or founder']
seniority_mapping = {level: i for i, level in enumerate(seniority_order)}

print("序数变量 seniority 的编码映射:")
for level, code in seniority_mapping.items():
    print(f"  {level} -> {code}")

# 检查是否有未知的 seniority 值
unknown_seniority = set(final_df['seniority'].dropna().unique()) - set(seniority_order)
if unknown_seniority:
    print(f"⚠️ 发现未知的 seniority 值: {unknown_seniority}")

# 创建序数编码的 seniority
final_df['seniority_ordinal'] = final_df['seniority'].map(seniority_mapping)

# 预处理：去除极端缺失
work = final_df.dropna(subset=['worker_id','year','seniority','rarity_score', 'seniority_ordinal']).copy()
print(f"去除缺失后: {len(work)} 行")

# 确保 year 是整数
work['year'] = work['year'].astype(int)
work['worker_id'] = work['worker_id'].astype(str).str.strip()

# 创建时间滞后数据
print("创建时间滞后变量...")

# 按 worker_id 分组，为每个人创建滞后变量
lag_data = []

for worker_id, group in work.groupby('worker_id'):
    # 按年份排序
    group_sorted = group.sort_values('year')
    
    # 检查年份是否连续（至少有两个连续年份）
    years = group_sorted['year'].values
    
    for i in range(len(group_sorted) - 1):
        current_row = group_sorted.iloc[i]
        next_row = group_sorted.iloc[i + 1]
        
        # 检查是否是连续年份
        if next_row['year'] == current_row['year'] + 1:
            # 创建滞后观测：t年的解释变量 + t+1年的因变量
            lag_row = current_row.copy()  # t年的所有变量作为解释变量
            lag_row['seniority_next'] = next_row['seniority']  # t+1年的seniority作为因变量
            lag_row['seniority_ordinal_next'] = next_row['seniority_ordinal']  # 编码后的
            lag_row['year_current'] = current_row['year']  # 当前年份
            lag_row['year_next'] = next_row['year']  # 下一年份
            
            lag_data.append(lag_row)

# 转换为DataFrame
lag_df = pd.DataFrame(lag_data)
print(f"时间滞后数据集: {len(lag_df)} 行 (从 {len(work)} 行创建)")

if len(lag_df) == 0:
    print("❌ 没有找到连续年份的数据，无法创建滞后数据集")
    exit()

# 显示数据统计
print(f"\n数据分布:")
print(f"- 年份范围: {lag_df['year_current'].min()}-{lag_df['year_current'].max()} -> {lag_df['year_next'].min()}-{lag_df['year_next'].max()}")
print(f"- 独特个体数: {lag_df['worker_id'].nunique()}")
print(f"- 国家分布: {lag_df['country'].value_counts().to_dict()}")

# 处理性别变量
if 'gender' in lag_df.columns:
    lag_df['gender'] = lag_df['gender'].astype(str).str.strip().str.lower()
    lag_df['gender_female'] = lag_df['gender'].map({'female': 1.0, 'male': 0.0})
else:
    lag_df['gender_female'] = np.nan

# 控制变量集合（排除主键与因变量/主解释变量）
exclude = set(['worker_id','country','year','seniority','rarity_score', 'gender', 
               'seniority_next', 'seniority_ordinal_next', 'year_current', 'year_next',
               'seniority_ordinal'])
control_cols_all = [c for c in lag_df.columns if c not in exclude]

# 分类变量（object 或 category）与数值变量
cat_controls = [c for c in control_cols_all if str(lag_df[c].dtype) in ['object','category']]
num_controls = [c for c in control_cols_all if c not in cat_controls]

print(f"\n控制变量:")
print(f"- 数值控制变量: {num_controls}")
print(f"- 分类控制变量: {cat_controls}")

countries = list(pd.unique(lag_df['country']))
print(f"- 分析国家: {countries}")

results = []
models = {}

print(f"\n开始分国家回归分析...")

for ctry in countries:
    print(f"\n--- 分析国家: {ctry} ---")
    dfc = lag_df[lag_df['country'] == ctry].copy()
    if dfc.empty:
        print(f"  {ctry}: 无数据，跳过")
        continue

    print(f"  {ctry}: 数据量 {len(dfc)} 行，个体数 {dfc['worker_id'].nunique()}")

    # 构建设计矩阵：rarity_score + controls（类别控制独热编码）
    X_parts = [dfc[['rarity_score']]]

    # 数值控制
    if num_controls:
        num_data = dfc[num_controls]
        if not num_data.empty:
            X_parts.append(num_data)

    # 类别控制做独热编码（drop_first 避免完全共线）
    if cat_controls:
        available_cats = [c for c in cat_controls if c in dfc.columns]
        if available_cats:
            dummies = pd.get_dummies(dfc[available_cats], drop_first=True, dtype=float)
            if not dummies.empty:
                X_parts.append(dummies)

    X = pd.concat(X_parts, axis=1)
    y_ordinal = dfc['seniority_ordinal_next']  # 使用下一年的序数编码seniority
    y_original = dfc['seniority_next']  # 保留原始文本用于参考

    # 去除任何仍有缺失的行
    valid = X.notna().all(axis=1) & y_ordinal.notna()
    X = X.loc[valid]
    y_ordinal = y_ordinal.loc[valid]
    y_original = y_original.loc[valid]
    dfc_valid = dfc.loc[valid]

    print(f"  {ctry}: 清理后剩余 {len(X)} 观测值")

    if len(X) < 50:  # 最小样本量检查
        print(f"  {ctry}: 样本量太小(<50)，跳过")
        continue

    # 方法1: 尝试序数回归（Ordered Logit）
    ordered_success = False
    
    if HAS_ORDERED:
        try:
            # 确保所有数据都是数值型
            X_clean = X.copy()
            
            # 检查并转换所有列为数值型
            for col in X_clean.columns:
                if X_clean[col].dtype == 'object':
                    X_clean[col] = pd.to_numeric(X_clean[col], errors='coerce')
                elif X_clean[col].dtype == 'bool':
                    X_clean[col] = X_clean[col].astype(float)
            
            # 去除转换后产生的NaN行
            mask = X_clean.notna().all(axis=1) & y_ordinal.notna()
            X_clean = X_clean.loc[mask]
            y_clean = y_ordinal.loc[mask].astype(int)
            
            # 确保所有变量都是数值型（float或int）
            for col in X_clean.columns:
                if X_clean[col].dtype.kind not in 'biufc':
                    X_clean[col] = X_clean[col].astype(float)
            
            print(f"  {ctry}: 序数回归数据准备完成，{len(y_clean)} 观测值")
            print(f"  因变量分布: {dict(pd.Series(y_clean).value_counts().sort_index())}")
            
            # OrderedModel 不需要手动添加常数项
            X_final = X_clean
            
            # 使用 statsmodels 的序数回归
            mod = OrderedModel(y_clean, X_final, distr='logit')
            res = mod.fit(disp=False)
            models[(ctry,'ordered_logit_lag')] = res

            coef = res.params.get('rarity_score', np.nan)
            se = res.bse.get('rarity_score', np.nan)
            pval = res.pvalues.get('rarity_score', np.nan)
            pseudo_r2 = getattr(res, 'prsquared', np.nan)
            n = int(res.nobs)
            results.append({'country': ctry, 'method': 'OrderedLogit_Lag', 'coef_rarity': coef, 'se': se, 'pval': pval, 'PseudoR2': pseudo_r2, 'N': n})
            ordered_success = True
            print(f"✓ {ctry}: 时间滞后序数回归成功，系数={coef:.4f}, p值={pval:.4f}")
            
        except Exception as e:
            print(f"⚠ {ctry}: 序数回归失败: {str(e)}")
            ordered_success = False
    
    # 方法2: 如果序数回归失败，尝试简化模型
    if not ordered_success:
        print(f"  {ctry}: 尝试简化的时间滞后序数回归...")
        
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
            models[(ctry,'ordered_logit_lag_simple')] = res_simple

            coef = res_simple.params.get('rarity_score', np.nan)
            se = res_simple.bse.get('rarity_score', np.nan)
            pval = res_simple.pvalues.get('rarity_score', np.nan)
            pseudo_r2 = getattr(res_simple, 'prsquared', np.nan)
            n = int(res_simple.nobs)
            results.append({'country': ctry, 'method': 'OrderedLogit_Lag_Simple', 'coef_rarity': coef, 'se': se, 'pval': pval, 'PseudoR2': pseudo_r2, 'N': n})
            print(f"✓ {ctry}: 简化时间滞后序数回归成功，系数={coef:.4f}, p值={pval:.4f}")
            
        except Exception as e2:
            print(f"✗ {ctry}: 所有时间滞后序数回归方法都失败了: {str(e2)}")
            results.append({'country': ctry, 'method': 'FAILED', 'coef_rarity': np.nan, 'se': np.nan, 'pval': np.nan, 'PseudoR2': np.nan, 'N': 0})

lag_results_summary = pd.DataFrame(results).sort_values(['country']).reset_index(drop=True)
print('\n' + '='*60)
print('时间滞后回归结果：t年rarity_score预测t+1年seniority')
print('='*60)
print(lag_results_summary)

# === 将结果写入一个 txt 报告 ===
import datetime

report_path = os.path.join(os.path.dirname(__file__), "time_lag_regression_report.txt")
ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

with open(report_path, "w", encoding="utf-8") as f:
    f.write(f"Time Lag Ordinal Regression Report\nGenerated: {ts}\n\n")
    f.write("RESEARCH QUESTION: Does t-year rarity_score predict t+1 year seniority?\n\n")
    f.write("IMPORTANT: seniority is an ordinal variable (Junior < Assistant < Regular < Senior < Leader < Chief or founder)\n\n")
    f.write("Statistical Methods Used:\n")
    f.write("1. Primary: Ordered Logit regression with time lag\n")
    f.write("2. Fallback: Simplified Ordered Logit with essential controls only\n\n")
    f.write("Time Lag Structure:\n")
    f.write("- Explanatory variables (X): t-year data (rarity_score, controls)\n")
    f.write("- Outcome variable (Y): t+1-year seniority\n")
    f.write("- Note: Final year observations are dropped (no t+1 data available)\n\n")
    f.write("Controls used:\n")
    f.write(f"- Numeric controls: {num_controls}\n")
    f.write(f"- Categorical controls (one-hot): {cat_controls}\n\n")
    f.write("Seniority encoding:\n")
    for level, code in seniority_mapping.items():
        f.write(f"- {level} -> {code}\n")
    f.write("\nNotes:\n")
    f.write("- gender encoded as gender_female: female=1, male=0; other/unknown as NaN\n")
    f.write("- Ordered regression is appropriate for ordinal outcomes\n")
    f.write("- Coefficients represent log-odds of being in higher seniority category\n")
    f.write("- Positive coefficient: higher rarity predicts higher future seniority\n")
    f.write("- Negative coefficient: higher rarity predicts lower future seniority\n\n")

    f.write("Country-level summary (one row per country):\n")
    f.write(lag_results_summary.to_string(index=False))
    f.write("\n\n")

    for ctry in countries:
        # 寻找该国家可用的模型
        key = None
        if (ctry, 'ordered_logit_lag') in models:
            key = (ctry, 'ordered_logit_lag')
        elif (ctry, 'ordered_logit_lag_simple') in models:
            key = (ctry, 'ordered_logit_lag_simple')
        
        if key is None:
            f.write(f"Country: {ctry} - 时间滞后序数回归失败，无可用模型\n\n")
            continue
            
        f.write("=" * 80 + "\n")
        f.write(f"Country: {ctry} | Model: {key[1]}\n")
        f.write("=" * 80 + "\n")
        f.write(str(models[key].summary()))
        f.write("\n\n")

print(f"\n完整时间滞后回归报告已写入: {report_path}")

# 比较同期与滞后效应
print(f"\n" + "="*60)
print("分析总结:")
print("="*60)
print("时间滞后分析帮助我们理解:")
print("1. t年的rarity_score是否能预测t+1年的职业发展")
print("2. 这种预测效应在不同国家是否不同")
print("3. 相比同期效应，滞后效应的大小和方向")
print("\n建议对比前一个分析（03_fixed_effects_model.py）的同期效应!")
