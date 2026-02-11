



dissindic <- function(diss, group, gower=FALSE, squared=FALSE, weights=NULL){
	dissmfacwresid <- function(formula, data) {
		
			#To ensure dissmatrix can be a dist object as well
			dissmatrix <- as.matrix(eval(formula[[2]], data, parent.frame()))
			formula[[2]] <- NULL
			#Model matrix from forumla
			predictor_frame <- model.frame(formula, data, na.action=na.pass, drop.unused.levels = TRUE)
			seqok <- complete.cases(predictor_frame)
			#To make sure unused levels after removing NA values are actually removed
			predictor_frame <- model.frame(formula, data[seqok, ], drop.unused.levels = TRUE)
			predictor_terms <- attr(predictor_frame, "terms")
			predictor <- model.matrix(formula, predictor_frame)
			if (is.null(weights)) {
				weights <- rep(1, sum(seqok))
			} else {
				weights <- weights[seqok]
			}
			
			dissmatrix <- dissmatrix[seqok, seqok]

			if (!gower) {
				g_matrix <- gower_matrix(dissmatrix, squared, weights=weights)

			} else {
				g_matrix <- dissmatrix
			}
			# n <- nrow(g_matrix)
			var_list <- attr(predictor, "assign")
			W_sqrt <- sqrt(weights)
			W_sqrt_mat <- tcrossprod(W_sqrt)
			# hat_matrix_qr <- function(pred) {
				# qr_matrix <- qr(W_sqrt*pred)
				# q_matrix <- qr.Q(qr_matrix)
				# hat_matrix <- tcrossprod(q_matrix)
			# }
			hatw_matrix_qr <- function(pred) {
				qr_matrix <- qr(W_sqrt*pred)
				q_matrix <- qr.Q(qr_matrix)
				hat_matrix <- tcrossprod(q_matrix)
				return(W_sqrt_mat*hat_matrix)
			}
			var_list_index <- (1:length(var_list))
			#List of variable
			var_names <- unique(var_list)
			#Number of variable (minus cte)
			nterms <- length(var_names) - 1
			
			W_mat <- matrix(0, nrow=nrow(g_matrix), ncol=ncol(g_matrix))
			diag(W_mat) <- weights
			residualsList <- matrix(0, ncol=(nterms+2), nrow=length(weights))
			colnames(residualsList) <- c(attr(predictor_terms, "term.labels"), "NullModels", "Complete")
			#Compute all  "backward" SCexp based on QR decomposition
			#for (var in 1:(nterms)) {
			#	pred <- predictor[, c(var_list_index[var_list!=var])]
			#	hwm <- hatw_matrix_qr(pred)
				##No need to transpose, G is symmetric
			#	residualsList[, var] <- diag((W_mat-hwm) %*% g_matrix)/weights
				
			#}
			
			residualsList[, "NullModels"] <- diag(g_matrix)
			hwm <- hatw_matrix_qr(predictor)
			residualsList[, "Complete"] <- diag((W_mat-hwm) %*% g_matrix)/weights
			
			
			return(list(indic=as.data.frame(residualsList, check.names=FALSE), complete=seqok))
	}
	if(length(group)!=nrow(diss)||length(group)!=ncol(diss)){
		message(" [!] diss should be a squared matrix with one row/column per observation and group a variable with one value per observation.\n")
	}
	xx <- dissmfacwresid(diss~group, data=data.frame(group=group))
	if(any(!xx$complete)){
		message(" [!] Missing values in group results in missing values in the computed indicators.\n")
	}
	ret <- data.frame(group=group, marginality=NA_real_, gain=NA_real_ )
	ret$marginality[xx$complete] <- xx$indic$Complete
	ret$gain[xx$complete] <- xx$indic$NullModels- xx$indic$Complete
	
	return(ret)
}
