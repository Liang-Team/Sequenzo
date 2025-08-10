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
import warnings
from statsmodels.tools.sm_exceptions import ConvergenceWarning

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

# 对关键解释变量 rarity_score 做 z-score 标准化（按国家分组），以便跨国比较（每 +1 SD）
def _zscore_series(s: pd.Series) -> pd.Series:
    m = s.mean()
    sd = s.std(ddof=0)
    if pd.notna(sd) and sd > 0:
        return (s - m) / sd
    # 若标准差为0或NaN，则返回0，避免产生NaN/Inf
    return pd.Series(0.0, index=s.index)

lag_df['rarity_score_z'] = lag_df.groupby('country', group_keys=False)['rarity_score'].apply(_zscore_series)
print("已对 rarity_score 做按国家的 z-score 标准化（1 单位 = 1 个国家内标准差）")

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
exclude = set([
    'worker_id','country','year','seniority','gender',
    'seniority_next', 'seniority_ordinal_next', 'year_current', 'year_next',
    'seniority_ordinal',
    'rarity_score', 'rarity_score_z'  # 显式排除主解释变量，避免重复加入
])
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

# === 工具函数：计算有序Logit的边际效应（APE） ===
def compute_ordered_marginal_effects(results_obj, X_df, var_name, seniority_levels):
    """
    计算有序Logit模型中某个变量对每个类别概率的平均偏效应（Average Partial Effects）。
    优先使用 statsmodels 的 get_margeff；若失败，则使用数值近似。

    返回：
      {
        'levels': [level names...],
        'effects': np.ndarray shape (J,),  # 各类别的APE
        'se': np.ndarray shape (J,) or None,  # 如果可得，返回标准误
        'summary': str  # 可读性摘要
      }
    """
    summary_text = ""
    try:
        # 1) 优先使用内置margins
        me = results_obj.get_margeff(at='overall')
        summary_text = str(me.summary())

        # 获取变量索引
        exog_names = list(results_obj.model.exog_names)
        if var_name not in exog_names:
            # 兼容不同命名（比如经过哑变量）
            # 尝试找到第一个包含该前缀的列
            candidates = [i for i, n in enumerate(exog_names) if n == var_name or n.endswith(f"[{var_name}]") or n.startswith(var_name)]
            if not candidates:
                raise ValueError(f"变量 {var_name} 不在模型自变量中: {exog_names}")
            var_idx = candidates[0]
        else:
            var_idx = exog_names.index(var_name)

        me_vals = me.margeff
        me_se = getattr(me, 'margeff_se', None)

        # 形状自适配
        def pick_axis(arr):
            if arr is None:
                return None
            if arr.ndim == 1:
                return arr
            if arr.shape[0] == len(exog_names):
                return arr[var_idx, :]
            if arr.shape[-1] == len(exog_names):
                return arr[:, var_idx]
            # 兜底：尝试第一维
            return arr[var_idx]

        eff = pick_axis(me_vals)
        se = pick_axis(me_se) if me_se is not None else None

        # 截取到类别数长度
        J = len(seniority_levels)
        eff = np.asarray(eff).reshape(-1)[:J]
        if se is not None:
            se = np.asarray(se).reshape(-1)[:J]

        return {
            'levels': seniority_levels,
            'effects': eff,
            'se': se,
            'summary': summary_text
        }
    except Exception as e:
        # 2) 数值近似：平均 (P(y=j|x+dx) - P(y=j|x)) / dx
        try:
            dx = 1e-4
            # 采样以加速
            max_sample = 5000
            if len(X_df) > max_sample:
                X_s = X_df.sample(n=max_sample, random_state=123)
            else:
                X_s = X_df

            if var_name not in X_s.columns:
                raise ValueError(f"变量 {var_name} 不在设计矩阵中")

            # 判断是否二元变量（0/1）
            col = pd.Series(pd.unique(X_s[var_name].dropna()))
            is_binary = set(np.sort(col.values)) <= {0, 1}

            if is_binary:
                # 离散变化：将该列整体从0切换到1，计算概率差的样本平均
                X0 = X_s.copy(); X0[var_name] = 0.0
                X1 = X_s.copy(); X1[var_name] = 1.0
                P0 = results_obj.model.predict(results_obj.params, exog=X0)
                P1 = results_obj.model.predict(results_obj.params, exog=X1)
                dP = (P1 - P0)
                ape = dP.mean(axis=0)
                J = len(seniority_levels)
                ape = np.asarray(ape).reshape(-1)[:J]
                return {
                    'levels': seniority_levels,
                    'effects': ape,
                    'se': None,
                    'summary': 'Discrete APE (0→1) over all observations'
                }

            # 连续变量：数值导数近似
            X_plus = X_s.copy()
            X_plus[var_name] = X_plus[var_name].astype(float) + dx

            # 预测概率矩阵 N x J
            P = results_obj.model.predict(results_obj.params, exog=X_s)
            P_plus = results_obj.model.predict(results_obj.params, exog=X_plus)

            # 平均偏效应
            dP = (P_plus - P) / dx
            ape = dP.mean(axis=0)

            J = len(seniority_levels)
            ape = np.asarray(ape).reshape(-1)[:J]

            return {
                'levels': seniority_levels,
                'effects': ape,
                'se': None,
                'summary': f"Numerical APE (dx={dx}) computed over {len(X_s)} samples"
            }
        except Exception as e2:
            return {
                'levels': seniority_levels,
                'effects': None,
                'se': None,
                'summary': f"边际效应计算失败: {e}; 近似也失败: {e2}"
            }

# === 工具函数：系数的平均边际效应（基于期望职级 E[Y]）===
def compute_ame_expected(results_obj, X_df, var_name):
    """
    返回单个数值：对期望职级 E[Y] 的平均边际效应。
    - 连续变量：数值导数 (E[Y|x+dx]-E[Y|x])/dx 的样本平均
    - 二元变量（0/1）：离散变化 E[Y|x=1]-E[Y|x=0] 的样本平均
    """
    # 采样以加速
    max_sample = 5000
    if len(X_df) > max_sample:
        X_s = X_df.sample(n=max_sample, random_state=123)
    else:
        X_s = X_df.copy()

    if var_name not in X_s.columns:
        return np.nan, 'missing'

    # 预测各类别概率
    def expected_y(exog):
        P = results_obj.model.predict(results_obj.params, exog=exog)
        levels = np.arange(P.shape[1])  # 0..J-1
        return (P * levels.reshape(1, -1)).sum(axis=1)

    col = X_s[var_name]
    # 判断是否二元
    unique_vals = pd.Series(pd.unique(col.dropna()))
    is_binary = set(np.sort(unique_vals.values)) <= {0, 1}

    if is_binary:
        X0 = X_s.copy(); X0[var_name] = 0.0
        X1 = X_s.copy(); X1[var_name] = 1.0
        Ey0 = expected_y(X0)
        Ey1 = expected_y(X1)
        ame = float((Ey1 - Ey0).mean())
        return ame, 'discrete_0to1'
    else:
        dx = 1e-4
        X_plus = X_s.copy(); X_plus[var_name] = X_plus[var_name].astype(float) + dx
        Ey = expected_y(X_s)
        Ey_plus = expected_y(X_plus)
        ame = float(((Ey_plus - Ey) / dx).mean())
        return ame, 'derivative'

print(f"\n开始分国家回归分析...")

for ctry in countries:
    print(f"\n--- 分析国家: {ctry} ---")
    dfc = lag_df[lag_df['country'] == ctry].copy()
    if dfc.empty:
        print(f"  {ctry}: 无数据，跳过")
        continue

    print(f"  {ctry}: 数据量 {len(dfc)} 行，个体数 {dfc['worker_id'].nunique()}")

    # 构建设计矩阵：rarity_score_z + controls（类别控制独热编码）
    X_parts = [dfc[['rarity_score_z']]]

    # 数值控制
    if num_controls:
        num_data = dfc[num_controls]
        if not num_data.empty:
            X_parts.append(num_data)

    # 类别控制做独热编码（指定参考组，drop_first 删除参考组列）
    if cat_controls:
        available_cats = [c for c in cat_controls if c in dfc.columns]
        if available_cats:
            df_cat = dfc[available_cats].copy()
            # 指定参考组顺序
            if 'highest_educational_degree' in df_cat.columns:
                vals = list(pd.Series(df_cat['highest_educational_degree'].astype(str).unique()))
                vals = [v for v in vals if v != 'Bachelor']
                ordered = ['Bachelor'] + vals
                df_cat['highest_educational_degree'] = pd.Categorical(
                    df_cat['highest_educational_degree'], categories=ordered, ordered=True
                )
            if 'internationalization' in df_cat.columns:
                vals = list(pd.Series(df_cat['internationalization'].astype(str).unique()))
                vals = [v for v in vals if v != 'Local']
                ordered = ['Local'] + vals
                df_cat['internationalization'] = pd.Categorical(
                    df_cat['internationalization'], categories=ordered, ordered=True
                )
            if 'simplified_company_size' in df_cat.columns:
                ref = 'Micro (0-10 employees)'
                vals = list(pd.Series(df_cat['simplified_company_size'].astype(str).unique()))
                vals = [v for v in vals if v != ref]
                ordered = [ref] + vals
                df_cat['simplified_company_size'] = pd.Categorical(
                    df_cat['simplified_company_size'], categories=ordered, ordered=True
                )

            dummies = pd.get_dummies(df_cat, drop_first=True, dtype=float)
            if not dummies.empty:
                X_parts.append(dummies)

    # === 加入时间固定效应（Year FE：year_current 的虚拟变量）===
    if 'year_current' in dfc.columns:
        year_cat = dfc['year_current'].astype(int).astype('category')
        year_dummies = pd.get_dummies(year_cat, drop_first=True, prefix='year', dtype=float)
        if not year_dummies.empty:
            X_parts.append(year_dummies)

    X = pd.concat(X_parts, axis=1)
    # 去重列名，防止重复列导致后续 dtype 检查把列切成 DataFrame
    if X.columns.duplicated().any():
        X = X.loc[:, ~X.columns.duplicated()]
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
            
            # 检查并转换所有列为数值型；同时消除完美多重共线（零方差）
            for col in list(X_clean.columns):
                s = X_clean[col]
                if pd.api.types.is_object_dtype(s):
                    X_clean[col] = pd.to_numeric(s, errors='coerce')
                elif pd.api.types.is_bool_dtype(s):
                    X_clean[col] = s.astype(float)
                # 去除常数列（零方差）
                if X_clean[col].nunique(dropna=False) <= 1:
                    X_clean.drop(columns=[col], inplace=True)
            
            # 去除转换后产生的NaN行
            mask = X_clean.notna().all(axis=1) & y_ordinal.notna()
            X_clean = X_clean.loc[mask]
            y_clean = y_ordinal.loc[mask].astype(int)
            
            # 确保所有变量都是数值型（float或int）
            for col in list(X_clean.columns):
                s = X_clean[col]
                if not pd.api.types.is_numeric_dtype(s):
                    X_clean[col] = s.astype(float)
            
            print(f"  {ctry}: 序数回归数据准备完成，{len(y_clean)} 观测值")
            print(f"  因变量分布: {dict(pd.Series(y_clean).value_counts().sort_index())}")
            
            # OrderedModel 不需要手动添加常数项
            X_final = X_clean
            
            # 使用 statsmodels 的序数回归
            mod = OrderedModel(y_clean, X_final, distr='logit')
            # 首选 L-BFGS 以提升大样本/高维时的收敛稳定性
            with warnings.catch_warnings(record=True) as wlist:
                warnings.simplefilter('always', ConvergenceWarning)
                res = mod.fit(method='lbfgs', maxiter=2000, disp=False)
                # 如果仍出现收敛警告，尝试备用优化器
                if any(getattr(w, 'category', None) is ConvergenceWarning for w in wlist):
                    print("  警告: L-BFGS 未收敛，改用 BFGS 并提高迭代上限重试...")
                    res = mod.fit(method='bfgs', maxiter=4000, disp=False)
            # 聚类稳健标准误（按个体）
            try:
                res = res.get_robustcov_results(cov_type='cluster', groups=dfc_valid.loc[X_final.index, 'worker_id'])
            except Exception as _:
                pass
            models[(ctry,'ordered_logit_lag')] = res

            coef = res.params.get('rarity_score_z', np.nan)
            se = res.bse.get('rarity_score_z', np.nan)
            pval = res.pvalues.get('rarity_score_z', np.nan)
            pseudo_r2 = getattr(res, 'prsquared', np.nan)
            n = int(res.nobs)
            # 计算期望职级的平均边际效应（AME）
            ame_rarity, ame_type = compute_ame_expected(res, X_final, 'rarity_score_z')
            models[(ctry,'ordered_logit_lag_ame_rarity')] = {'ame': ame_rarity, 'type': ame_type}

            results.append({
                'country': ctry,
                'method': 'OrderedLogit_Lag',
                'coef_rarity_z': coef,
                'se': se,
                'pval': pval,
                'PseudoR2': pseudo_r2,
                'N': n,
                'AME_rarity_per1SD': ame_rarity
            })
            ordered_success = True
            print(f"✓ {ctry}: 时间滞后序数回归成功（rarity_score 标准化，1单位=1SD），系数={coef:.4f}, p值={pval:.4f}, AME(E[Y])={ame_rarity:.4f} per +1SD")
            
            # === 计算每个变量对各类别概率的 APE（边际效应） ===
            seniority_levels = seniority_order  # 使用实际的级别名称
            per_var_ape = {}
            
            exog_cols = list(X_final.columns)  # 模型里的所有解释变量
            print(f"  {ctry}: 计算所有变量的边际效应，变量数: {len(exog_cols)}")
            
            for v in exog_cols:
                try:
                    ape_obj = compute_ordered_marginal_effects(res, X_final, v, seniority_levels)
                    # ape_obj['effects'] 是长度为6的数组：对 P(Y=0..5) 的 APE
                    per_var_ape[v] = {
                        "effects": ape_obj["effects"],   # np.ndarray, shape (6,)
                        "se": ape_obj["se"],            # 若 statsmodels 计算成功会有 SE，否则 None
                        "note": ape_obj["summary"]      # 说明是 analytical 还是 numerical
                    }
                except Exception as ape_e:
                    per_var_ape[v] = {"effects": None, "se": None, "note": f"APE failed: {ape_e}"}
            
            models[(ctry, 'ordered_logit_lag_ape_allvars')] = per_var_ape
            print(f"  {ctry}: 边际效应计算完成")
            
        except Exception as e:
            print(f"⚠ {ctry}: 序数回归失败: {str(e)}")
            ordered_success = False
    
    # 方法2: 如果序数回归失败，尝试简化模型
    if not ordered_success:
        print(f"  {ctry}: 尝试简化的时间滞后序数回归...")
        
        try:
            # 只使用 rarity_score_z 和几个主要控制变量
            essential_controls = []
            if 'work_years' in num_controls:
                essential_controls.append('work_years')
            if 'gender_female' in num_controls:
                essential_controls.append('gender_female')
            
            X_simple = dfc[['rarity_score_z'] + essential_controls].copy()

            # 加入时间固定效应到简化模型（Year FE）
            if 'year_current' in dfc.columns:
                year_cat = dfc['year_current'].astype(int).astype('category')
                year_dummies = pd.get_dummies(year_cat, drop_first=True, prefix='year', dtype=float)
                if not year_dummies.empty:
                    X_simple = pd.concat([X_simple, year_dummies], axis=1)
            # 去重列名
            if X_simple.columns.duplicated().any():
                X_simple = X_simple.loc[:, ~X_simple.columns.duplicated()]
            
            # 确保数据类型正确
            for col in list(X_simple.columns):
                s = X_simple[col]
                if pd.api.types.is_object_dtype(s):
                    X_simple[col] = pd.to_numeric(s, errors='coerce')
                elif pd.api.types.is_bool_dtype(s):
                    X_simple[col] = s.astype(float)
                if X_simple[col].nunique(dropna=False) <= 1:
                    X_simple.drop(columns=[col], inplace=True)
            
            # 清理数据
            mask = X_simple.notna().all(axis=1) & y_ordinal.notna()
            X_simple_clean = X_simple.loc[mask]
            y_simple_clean = y_ordinal.loc[mask].astype(int)
            
            # 确保所有数据类型都是数值型
            for col in list(X_simple_clean.columns):
                s = X_simple_clean[col]
                if not pd.api.types.is_numeric_dtype(s):
                    X_simple_clean[col] = s.astype(float)
            
            print(f"  {ctry}: 简化模型，变量: {list(X_simple_clean.columns)}, 观测值: {len(y_simple_clean)}")
            
            # 序数回归（不添加常数项）
            mod_simple = OrderedModel(y_simple_clean, X_simple_clean, distr='logit')
            with warnings.catch_warnings(record=True) as wlist:
                warnings.simplefilter('always', ConvergenceWarning)
                res_simple = mod_simple.fit(method='lbfgs', maxiter=2000, disp=False)
                if any(getattr(w, 'category', None) is ConvergenceWarning for w in wlist):
                    print("  警告: 简化模型 L-BFGS 未收敛，改用 BFGS 并提高迭代上限重试...")
                    res_simple = mod_simple.fit(method='bfgs', maxiter=4000, disp=False)
            # 简化模型：聚类稳健标准误（按个体）
            try:
                res_simple = res_simple.get_robustcov_results(cov_type='cluster', groups=dfc_valid.loc[X_simple_clean.index, 'worker_id'])
            except Exception as _:
                pass
            models[(ctry,'ordered_logit_lag_simple')] = res_simple

            coef = res_simple.params.get('rarity_score_z', np.nan)
            se = res_simple.bse.get('rarity_score_z', np.nan)
            pval = res_simple.pvalues.get('rarity_score_z', np.nan)
            pseudo_r2 = getattr(res_simple, 'prsquared', np.nan)
            n = int(res_simple.nobs)
            # 简化模型的期望值AME
            ame_rarity_s, ame_type_s = compute_ame_expected(res_simple, X_simple_clean, 'rarity_score_z')
            models[(ctry,'ordered_logit_lag_simple_ame_rarity')] = {'ame': ame_rarity_s, 'type': ame_type_s}

            results.append({
                'country': ctry,
                'method': 'OrderedLogit_Lag_Simple',
                'coef_rarity_z': coef,
                'se': se,
                'pval': pval,
                'PseudoR2': pseudo_r2,
                'N': n,
                'AME_rarity_per1SD': ame_rarity_s
            })
            print(f"✓ {ctry}: 简化时间滞后序数回归成功（rarity_score 标准化，1单位=1SD），系数={coef:.4f}, p值={pval:.4f}, AME(E[Y])={ame_rarity_s:.4f} per +1SD")
            
            # === 计算简化模型每个变量对各类别概率的 APE ===
            seniority_levels = seniority_order
            per_var_ape_simple = {}
            
            exog_cols_simple = list(X_simple_clean.columns)
            print(f"  {ctry}: 计算简化模型所有变量的边际效应，变量数: {len(exog_cols_simple)}")
            
            for v in exog_cols_simple:
                try:
                    ape_obj = compute_ordered_marginal_effects(res_simple, X_simple_clean, v, seniority_levels)
                    per_var_ape_simple[v] = {
                        "effects": ape_obj["effects"],
                        "se": ape_obj["se"],
                        "note": ape_obj["summary"]
                    }
                except Exception as ape_e:
                    per_var_ape_simple[v] = {"effects": None, "se": None, "note": f"APE failed: {ape_e}"}
            
            models[(ctry, 'ordered_logit_lag_simple_ape_allvars')] = per_var_ape_simple
            print(f"  {ctry}: 简化模型边际效应计算完成")
            
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
    f.write(f"- Categorical controls (one-hot): {cat_controls}\n")
    f.write("- rarity_score standardized to z-score within country (1 unit = +1 SD)\n\n")
    f.write("Seniority encoding:\n")
    for level, code in seniority_mapping.items():
        f.write(f"- {level} -> {code}\n")
    f.write("\nNotes:\n")
    f.write("- gender encoded as gender_female: female=1, male=0; other/unknown as NaN\n")
    f.write("- Ordered regression is appropriate for ordinal outcomes\n")
    f.write("- Coefficients represent log-odds of being in higher seniority category\n")
    f.write("- Positive coefficient: higher rarity predicts higher future seniority\n")
    f.write("- Negative coefficient: higher rarity predicts lower future seniority\n\n")

    # 固定效应与面板说明
    f.write("Fixed effects included:\n")
    f.write("- Year FE via dummies on year_current (drop_first)\n")
    f.write("- No individual FE; pooled across workers\n\n")

    f.write("Country-level summary (one row per country):\n")
    f.write(lag_results_summary.to_string(index=False))
    f.write("\n\n")

    # 写入各国“系数的平均边际效应（对期望职级E[Y]）”
    f.write("Average Marginal Effects (on expected seniority, E[Y]) by Country:\n\n")
    for ctry in countries:
        ame_key = (ctry, 'ordered_logit_lag_ame_rarity') if (ctry, 'ordered_logit_lag_ame_rarity') in models else (
                  (ctry, 'ordered_logit_lag_simple_ame_rarity') if (ctry, 'ordered_logit_lag_simple_ame_rarity') in models else None)
        if ame_key is None:
            f.write(f"Country: {ctry} - 无AME结果\n\n")
            continue
        ame_obj = models[ame_key]
        ame = ame_obj.get('ame', np.nan)
        ame_type = ame_obj.get('type', '')
        f.write(f"- {ctry}: AME(E[Y]) for rarity_score (per +1 SD) = {ame:.6f} ({ame_type})\n")
    f.write("\n")

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
        
        # 写入边际效应 (APE)
        ape_key = None
        if key[1] == 'ordered_logit_lag':
            ape_key = (ctry, 'ordered_logit_lag_ape_allvars')
        elif key[1] == 'ordered_logit_lag_simple':
            ape_key = (ctry, 'ordered_logit_lag_simple_ape_allvars')
        
        if ape_key and ape_key in models:
            f.write("Marginal Effects (APE) by Variable:\n")
            f.write("(Change in probability for each seniority level per 1-unit increase)\n")
            f.write("-" * 60 + "\n")
            per_var_ape = models[ape_key]
            
            for v, obj in per_var_ape.items():
                f.write(f"\nVariable: {v}\n")
                if obj["effects"] is None:
                    f.write(f"  APE calculation failed: {obj['note']}\n")
                    continue
                
                eff = obj["effects"]
                f.write("  Seniority Level Effects:\n")
                for j, level_name in enumerate(seniority_order):
                    if j < len(eff):
                        f.write(f"    {level_name} (level {j}): {eff[j]:.6f}\n")
                
                if obj["se"] is not None:
                    f.write("  (Standard errors available from analytical margins)\n")
                
                f.write(f"  Method: {obj['note']}\n")
            
            f.write("\n" + "-" * 60 + "\n")
        else:
            f.write("Marginal Effects (APE): Not computed for this model.\n\n")

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
