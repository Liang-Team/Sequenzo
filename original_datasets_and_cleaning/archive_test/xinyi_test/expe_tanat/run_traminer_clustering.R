

library(TraMineR)
library(cluster)
library(optparse)

# setwd("/home/tguyet/Publications/2026/IJCAI_Demo/exp_cmp")


option_list <- list(
  make_option(
    c("-i","--iteration"),
    type="integer",
    default = 0
  ),
  make_option(
    c("-c","--nb_clusters"),
    type="integer",
    default = 4
  ),
  make_option(
    c("-m","--metric"),
    type="character",
    default = "OM"
  ),
  make_option(
    c("-d","--dataset"),
    type="character",
    default = NULL
  )
)

parser <- OptionParser(option_list=option_list)
args <- parse_args(parser, positional_arguments=TRUE)

metric_name=args$options$`metric`
dataset_name=args$options$`dataset`

# 确定数据文件列表：-d 指定 generated_datasets 下的数据集名，否则使用位置参数
# 支持 -d output_n5000_l30 或 -d output_n5000_l30.csv（自动去除扩展名查找目录）
GENERATED_DATASETS_DIR <- "generated_datasets"
if (!is.null(dataset_name)) {
  dataset_dir <- file.path(GENERATED_DATASETS_DIR, dataset_name)
  if (!dir.exists(dataset_dir) && grepl("\\.", dataset_name)) {
    dataset_name_stem <- sub("\\.[^.]+$", "", dataset_name)
    dataset_dir <- file.path(GENERATED_DATASETS_DIR, dataset_name_stem)
  }
  if (!dir.exists(dataset_dir)) {
    stop(paste("error: dataset directory not found:", dataset_dir))
  }
  csv_files <- Sys.glob(file.path(dataset_dir, "*.csv"))
  dat_files <- Sys.glob(file.path(dataset_dir, "*.dat"))
  file_list <- sort(c(csv_files, dat_files))
  if (length(file_list) == 0) {
    stop(paste("error: no .csv or .dat files in", dataset_dir))
  }
} else {
  file_list <- args$args
  if (length(file_list) == 0) {
    message("用法: Rscript run_traminer_clustering.R -c <聚类数> -d <数据集名>")
    message("  或: Rscript run_traminer_clustering.R -c <聚类数> file1.csv [file2.csv ...]")
    quit(save = "no", status = 1)
  }
}

first <- T
max_ids=0

for (file in file_list) {
  data <- read.csv(file)
  data['stime']=data['stime']+1
  data['etime']=data['etime']+1
  data <- data[data['etime']>data['stime'],] #remove events with start date after end date
  
  data$id = data$id+max_ids
  max_ids = max(data$id)+1
  if (first) {
    alldata = data
    first <- F
  } else {
    alldata = rbind(alldata,data)
  }
}

start.time <- Sys.time()

seq <- seqdef(alldata, var = c("id", "stime", "etime", "event"), informat = "SPELL")

time.prep <- as.numeric(difftime(time1 = Sys.time(), time2 = start.time, units = "secs"))


# 一致性检验: OM, sm=CONSTANT cval=1 (Hamming: 0 match, 1 substitute), indel=1
# 与 TanaT EditSequenceMetric(entity_metric="hamming", indel_cost=1) 对应
scost <- seqsubm(seq, method = "CONSTANT", cval = 1, with.missing = TRUE)

start.time <- Sys.time()
dist <- seqdist(seq, method = "OM", indel = 1, sm = scost, with.missing = TRUE)
clusterward1 <- agnes(dist, diss = TRUE, method = "ward")
time.taken <- as.numeric(difftime(time1 = Sys.time(), time2 = start.time, units = "secs"))

# 输出距离矩阵（用于与 TanaT 一致性检验）
write.csv(as.matrix(dist), "dist_matrix_traminer.csv")
message("[TraMineR] 距离矩阵已保存至 dist_matrix_traminer.csv")

fileConn<-file("results_R.txt","a")
res <- paste(as.character(args$options$`iteration`),",TraMineR,",as.character(length(unique(alldata$id))),",",as.character(length(alphabet(seq))),",",as.character(nrow(alldata)/length(unique(alldata$id))),",",as.character(max(alldata['etime'])),",","OM",",",as.character(args$options$`nb_clusters`),",",as.character(time.prep),",",as.character(time.taken),",")
writeLines(res, fileConn)
close(fileConn)


# #############################
# library(yaml)
# library(rlang)
# 
# 
# 
# RESULT_FILE_NAME <- "results.csv"
# 
# if (!interactive()) {
#   args <- parse_benchmark_args()
#   config <- yaml.load_file(args$input_yaml)
#   data <- read.csv(args$input_csv)
#   
#   seq_obj <- seqdef(data = data[, -1])
#   seqdist_args <- c(list(seq_obj), config$sequence_metric)
#   dist_matrix <- do.call(seqdist, seqdist_args)
#   output_csv <- file.path(args$output_dir, RESULT_FILE_NAME)
#   write.csv(dist_matrix,output_csv)
#   print("TramineR - Sequence metric computation done.")
# }
