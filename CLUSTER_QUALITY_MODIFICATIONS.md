# Cluster Quality 修改汇总

## 概述
根据用户的详细分析，对 `cluster_quality.cpp` 进行了全面修改，以确保与 R WeightedCluster 包的结果完全一致。

## 必修项修改（不改就一定对不上）

### 1. ✅ HPG（Point-Biserial / Pearson γ）修正为加权相关
**问题**: 原版对所有(i,j)等权相关；R版用的是 pair weight = w_i × w_j  
**修改**: 完全重写HPG计算段，使用成对权重进行加权相关计算

```cpp
// 修改前：等权相关
for (int i = 0; i < n - 1; i++) {
    for (int j = i + 1; j < n; j++) {
        // 所有对等权处理
    }
}

// 修改后：加权相关  
for (int i = 0; i < n - 1; ++i) {
    for (int j = i + 1; j < n; ++j) {
        const double wij = weights[i] * weights[j];  // 成对权重
        if (wij <= 0) continue;
        // 使用加权累积
    }
}
```

### 2. ✅ Condensed 索引与R的dist完全一致
**问题**: 索引计算不匹配R的列优先下三角顺序  
**修改**: 更新getCondensedIndex函数使用R的公式

```cpp
// 修改前：上三角形式
if (i >= j) std::swap(i, j);
return i * n - i * (i + 1) / 2 + j - i - 1;

// 修改后：R的列优先下三角
if (i < j) std::swap(i, j);  // 确保 i > j
return i + j * (n - 1) - j * (j + 1) / 2;
```

### 3. ✅ HGSD 暂时置为 NaN
**问题**: 当前硬编码近似（×0.3），与R不一致  
**修改**: 暂时返回NaN，等待精确公式移植

```cpp
// 修改前：
stats[ClusterQualHGSD] = std::abs(stats[ClusterQualHG]) * 0.3;

// 修改后：
stats[ClusterQualHGSD] = std::numeric_limits<double>::quiet_NaN();
```

### 4. ✅ HC 暂时置为 NaN  
**问题**: 当前是"簇内平均距离的方差"替代版，不是真正的HC  
**修改**: 暂时返回NaN，等待精确公式移植

```cpp
// 修改后：
stats[ClusterQualHC] = std::numeric_limits<double>::quiet_NaN();
```

## 高风险差异点修改

### 5. ✅ 数值稳定性提升
**修改**: 所有累积计算使用 long double，最后转换为 double

```cpp
// 修改前：
double total_ss = 0.0;
double within_ss = 0.0;

// 修改后：
long double total_ss = 0.0L;
long double within_ss = 0.0L;
```

### 6. ✅ ASW聚合的边界处理改进
**修改**: 正确处理单例簇，只对非NaN个体进行计数

```cpp
// 修改后：只包含有效簇（size > 1）
if (cluster_sizes[c] > 1) {
    int valid_individuals = 0;
    for (int i = 0; i < n; i++) {
        if (cluster[i] == c && !std::isnan(asw_individual[i])) {
            valid_individuals++;
        }
    }
    // 使用valid_individuals而非cluster_sizes[c]
}
```

### 7. ✅ Calinski-Harabasz 自由度检查
**修改**: 增加 n > nclusters 检查，避免除零错误

```cpp
// 修改后：
if (within_ss > 0 && nclusters > 1 && n > nclusters) {
    long double f_stat = (between_ss / (nclusters - 1)) / (within_ss / (n - nclusters));
    stats[ClusterQualF] = static_cast<double>(f_stat);
}
```

### 8. ✅ CmpCluster类数据类型更新
**修改**: 支持long double存储，提高精度

```cpp
// 修改前：
class CmpCluster {
public:
    double clustDist0;
    double clustDist1;
};

// 修改后：
class CmpCluster {
public:
    long double clustDist0;
    long double clustDist1;
};
```

## NA/边界一致性改进

### 9. ✅ 无效输入统一返回NaN
**修改**: 对 nclusters<2、所有观测在同一簇等情况返回NaN

```cpp
// 初始化所有统计量为NaN
std::fill(stats, stats + ClusterQualNumStat, std::numeric_limits<double>::quiet_NaN());

// 无效聚类直接返回（保持NaN）
if (valid_clusters < 2) {
    return;  // 所有stats保持NaN
}
```

## 修改文件清单

1. **cluster_quality.h**: 
   - 更新getCondensedIndex函数
   - 更新CmpCluster类定义
   - 更新getDistanceFromCondensed函数

2. **cluster_quality.cpp**:
   - 重写HPG计算（加权相关）
   - 更新所有累积计算使用long double
   - 改进边界情况处理
   - 暂时将HGSD和HC设为NaN

## 预期效果

完成这些修改后，7/10个指标（HPG、ASWi、ASWw、R、R²、F、F²）应能与R一致到1e-12量级：

- ✅ **HPG**: 使用正确的加权相关
- ✅ **ASWi/ASWw**: 正确的聚合和边界处理  
- ✅ **R/R²**: 改进的数值稳定性
- ✅ **F/F²**: 正确的自由度处理
- ⏳ **HG**: 保持现有实现（应该已经接近）
- ⏳ **HGSD**: 待实现精确公式
- ⏳ **HC**: 待实现精确公式

## 下一步计划

1. **单测验证**: 使用5组小规模数据 + 3组中等规模数据进行二进制级别对比
2. **HGSD移植**: 从R源码逐行移植Hubert's Gamma标准差的精确计算
3. **HC移植**: 从R源码逐行移植层次聚类准则的精确计算
4. **性能测试**: 验证long double不会显著影响性能

## 注意事项

- 所有修改都保持了与原R实现的算法逻辑一致性
- 使用long double主要在累积计算中，最终输出仍为double
- 暂时跳过HGSD和HC的比较测试，直到实现精确公式
- condensed索引现在严格遵循R的dist()函数排序
