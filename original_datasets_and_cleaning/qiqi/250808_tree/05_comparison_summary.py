# 同期效应 vs 时间滞后效应对比分析
# 对比03_fixed_effects_model.py（同期）和04_time_lag_regression.py（滞后）的结果

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

print("="*80)
print("Rarity Score 效应对比：同期 vs 时间滞后")
print("="*80)

# 结果汇总
comparison_data = {
    'Country': ['美国', '印度'],
    'Contemporary_Effect': [-0.0122, 0.0261],  # 同期效应（03_fixed_effects_model.py）
    'Contemporary_PValue': [5.18e-11, 1.46e-27],
    'Contemporary_N': [198450, 127830],
    'Lag_Effect': [0.0106, 0.0982],  # 滞后效应（04_time_lag_regression.py）
    'Lag_PValue': [1e-6, 0.0],
    'Lag_N': [178605, 115047]
}

comparison_df = pd.DataFrame(comparison_data)

print("\n📊 效应大小对比:")
print("-" * 40)
for _, row in comparison_df.iterrows():
    country = row['Country']
    contemp = row['Contemporary_Effect']
    lag = row['Lag_Effect']
    
    print(f"\n{country}:")
    print(f"  同期效应:   {contemp:+.4f}")
    print(f"  滞后效应:   {lag:+.4f}")
    print(f"  方向变化:   {'是' if (contemp > 0) != (lag > 0) else '否'}")
    print(f"  效应增强:   {'是' if abs(lag) > abs(contemp) else '否'}")

print("\n" + "="*80)
print("重要发现:")
print("="*80)

print("\n🇺🇸 美国:")
print("  • 同期效应: NEGATIVE (-0.0122) - 稀有路径与较低职级相关")
print("  • 滞后效应: POSITIVE (+0.0106) - 稀有路径能预测未来更高职级")
print("  • 解释: 稀有路径在当下可能被低估，但长期来看是有价值的投资")

print("\n🇮🇳 印度:")
print("  • 同期效应: POSITIVE (+0.0261) - 稀有路径与较高职级相关")  
print("  • 滞后效应: POSITIVE (+0.0982, 更强) - 稀有路径强烈预测未来更高职级")
print("  • 解释: 稀有路径不仅当下有价值，而且是未来职业发展的强预测器")

print("\n📈 跨国对比:")
print("  • 两国的滞后效应都是正向的，说明稀有经历普遍有助于长期职业发展")
print("  • 印度的滞后效应 (0.0982) 比美国 (0.0106) 强很多")
print("  • 美国显示短期与长期效应的权衡，印度显示一致的正向效应")

print("\n🔍 理论含义:")
print("  1. 短期 vs 长期视角: 稀有路径的价值可能需要时间才能体现")
print("  2. 文化差异: 不同国家对非传统职业路径的认知和奖励机制不同")
print("  3. 投资视角: 在某些环境下，稀有经历是对未来的投资")

print("\n📋 建议进一步分析:")
print("  1. 按职业阶段分析：早期 vs 中后期职业发展")
print("  2. 按稀有程度分析：极稀有 vs 中等稀有路径")
print("  3. 按行业分析：不同行业对稀有经历的评价")
print("  4. 生存分析：稀有路径对职业转换概率的影响")

# 保存对比数据
comparison_df.to_csv('comparison_contemporary_vs_lag_effects.csv', index=False)
print(f"\n📁 对比数据已保存至: comparison_contemporary_vs_lag_effects.csv")

print("\n" + "="*80)
print("分析完成！")
print("="*80)
