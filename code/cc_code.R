# load in useful packages
library(readr)
library(dplyr)
library(DataExplorer)
library(pls)
library(glmnet)
library(corrplot)

# load data
CC <- read_csv("../data/crime_and_communities_data.csv")
CC <- as.data.frame(CC)

preview <- head(CC, 10)

# this is like a transposed version of print: columns run down the page, and data runs across. 
# this makes it possible to see every column in a data frame, essentially trying to show as much data as possible
glimpse(CC)

# summarize key attributes of the data
introduce(CC)

# the `plot_intro` function from the `DataExploration` package lets us 
# visualize our findings
plot_intro(CC)

# the following code breaks down the number of missing values by variable with a 
# for loop, looping over each coloumn in our dataset.  
na_per_col <- c()
for(i in 1:ncol(CC)){
  na_per_col[i] <- sum(is.na(CC[,i]))
}
names(na_per_col) <- names(CC)

# the `plot_missing` function from the `DataExploration` package allows us to 
# visualize the proportion of missing values
plot_missing(CC[,na_per_col!= 0])

cleaned_CC <- drop_columns(CC, names(na_per_col)[na_per_col == 1675])
cleaned_CC <- as.data.frame(cleaned_CC)
#dim(cleaned_CC)

par(mfrow=c(2,4))

# histograms
subset <- sample(names(cleaned_CC), 8)
for (name in subset){
  hist(cleaned_CC[ ,name], col ='gray80', main = paste(name), xlab ='', las = 1)
}

# correlation matrix for the above subset of predictors
corr <- cor(cleaned_CC[,subset])
corrplot(corr, type = "upper", order = "hclust", 
         tl.col = "black", tl.srt = 45)

# response vector
y <- cleaned_CC$ViolentCrimesPerPop
response <- "y"

# feature matrix with replaced missing value
X <- subset(cleaned_CC, select = -c(ViolentCrimesPerPop))
X$OtherPerCap[is.na(X$OtherPerCap)] <- mean(X$OtherPerCap, na.rm = TRUE)
predictors <- names(X)

# combine feature matrix and response vector for convenience
dat <- as.data.frame(cbind(X,y))

set.seed(1)
train_prop <- 0.6
total <- nrow(dat)

# training set
train <- sample(1:total, floor(total*train_prop))
testval <- c(1:total)[-train]

# validation set
validation <- sample(testval, size = floor(length(testval)/2))

# test set
test <- c(1:total)[-c(train, validation)]

# standardize feature matrix
X <- scale(X)

# compute variance-covariance matrix
n <- nrow(X)
S <- (1/(n-1)) * t(as.matrix(X)) %*% as.matrix(X)

# perform eigenvalue decomposition
EVD <- eigen(S)

# eigenvalues of S (the variances of each PC)
evalues <- EVD$values

# eigenvectors of S (loadings)
evectors <- EVD$vectors

# principal components (each PC is a linear combination of each of the original features)
pcs <- X %*% evectors
colnames(pcs) <- paste0('PC', 1:ncol(pcs))
#head(pcs[,1:4])

# high level summary of doing PCA
pc <- princomp(X, scores = TRUE)
summary(pc)

# plot of principal components
plot(pc, type = "l")

formula <- y ~. 
Xtrain <- scale(dat[train, c(predictors, response)])
Xtrain <- as.data.frame(Xtrain)

# PCR on training data
pcr_fit <- pcr(formula, data = Xtrain, scale = FALSE, ncomp = ncol(X))

PCR <- apply(pcr_fit$coefficients, MARGIN = 3, function(u) u)
matplot(t(PCR), type = "l", xlim = c(1, 102), ylim = c(-1, 1), ylab = "Value", xlab = "PCs")

# PLSR on training data
plsr_fit <-plsr(formula, data = Xtrain, scale = FALSE, ncomp = ncol(X))

PLSR <- apply(plsr_fit$coefficients, MARGIN = 3,function(u) u)
matplot(t(PLSR), type = "l", xlim = c(1, 102), ylim = c(-1.5, 1.25), ylab = "Value", xlab = "PLS Components")

# ridge regression on training data
x <- as.matrix(Xtrain[ ,predictors])
y <- Xtrain[ ,response]

ridge_fit <- glmnet(x, y, alpha = 0)
matplot(t(ridge_fit$beta), type = "l", las = 1, ylim = c(-0.1, 0.2), ylab = "Value", xlab = "Lambda")

# lasso regression on training data
lasso_fit <- glmnet(x, y, alpha = 1)
matplot(t(lasso_fit$beta), type = "l", las = 1, ylab = "Value", xlab = "Lambda")

# create folds
folds <- 5
if ((length(train)%%folds)==0){
  cuts <- seq(1,length(train), by = length(train)/folds)
} else {
  cuts <- seq(1,length(train), by = floor(length(train)/folds))
  cuts[folds+1] <-length(train)
}

# starting and ending positions of folds
fold_starts <- cuts[-length(cuts)]
fold_ends <- c(cuts[2:(length(cuts)-1)]-1, cuts[length(cuts)])

folds_list <-as.list(1:folds)
for(fold in 1:folds) {
  folds_list[[fold]] <- fold_starts[fold]:fold_ends[fold]
}

# the following code uses a for loop to do this process for PCR and PLSR. We also make use of 
# the `cv.glmnet` function to do this process for Ridge regression and Lasso regression.
mse_cv_pcr <- c()
mse_cv_plsr <- c()

# pcr and plsr
for(tune in 1:ncol(X)) {
  mse_pcr_aux <- c()
  mse_plsr_aux <- c()
  for(fold in 1:folds) {
    # train with k-1 folds
    Xtrain_folds <- scale(Xtrain[folds_list[[fold]], ])
    Xtrain_folds <- as.data.frame(Xtrain_folds)
    x_aux <- as.matrix(Xtrain_folds[, predictors])
    y_aux <- Xtrain_folds[,response]
    pcr_aux <- pcr(formula, data = Xtrain_folds, scale = FALSE, ncomp = tune)
    plsr_aux <-plsr(formula, data = Xtrain_folds, scale = FALSE, ncomp = tune)
    
    # predict holdout fold
    Xfoldout <- scale(Xtrain[unlist(folds_list[-fold]), predictors])
    Xfoldout <- as.data.frame(Xfoldout)
    yfoldout <- Xtrain[unlist(folds_list[-fold]), response]
    
    pcr_hat <- predict(pcr_aux, Xfoldout, ncomp = tune)
    plsr_hat <- predict(plsr_aux, Xfoldout, ncomp = tune)
    
    # measure performance
    mse_pcr_aux[fold] <- mean((pcr_hat[,,1]-yfoldout)^2)
    mse_plsr_aux[fold] <- mean((plsr_hat[,,1]-yfoldout)^2)
  }
  mse_cv_pcr[tune] <- mean(mse_pcr_aux)
  mse_cv_plsr[tune] <- mean(mse_plsr_aux)
}

lasso <- cv.glmnet(as.matrix(Xtrain[,c(1:102)]), Xtrain[,response], alpha = 1, type.measure = "mse", nfolds = 5)
ridge <- cv.glmnet(as.matrix(Xtrain[,c(1:102)]), Xtrain[,response], alpha = 0, type.measure = "mse", nfolds = 5)

# lasso
lasso_keep <- lasso$nzero[lasso$lambda == lasso$lambda.min]

# ridge
ridge_keep <- ridge$nzero[ridge$lambda == ridge$lambda.min]

# show the best tuning parameter for each method
cv_mse <- data.frame(pcr = mse_cv_pcr, plsr = mse_cv_plsr)
selected_pcr_plsr <- apply(cv_mse, 2, which.min)
keep <- c(selected_pcr_plsr, "ridge" = ridge$lambda.min, "lasso" = lasso$lambda.min)
keep

Xval <- scale(dat[validation, c(predictors, response)])
x_val <- as.matrix(Xval[,predictors])
y_val <- Xval[, response]

pcr_hat <- predict(pcr_fit, x_val, ncomp = keep[1])
plsr_hat <- predict(plsr_fit, x_val, ncomp = keep[2])
las_hat <- predict(lasso_fit, newx = x_val, s = keep[3])
rr_hat <- predict(ridge_fit, newx = x_val, s = keep[4])

mse_val <- c(
  "pcr_mse" = mean((pcr_hat[,,1]-y_val)^2),
  "plsr_mse" = mean((plsr_hat[,,1]-y_val)^2),
  "ridge_mse" = mean((rr_hat[,1]-y_val)^2),
  "lasso_mse" = mean((las_hat[,1]-y_val)^2)
)
mse_val

# test the performance of the best model
Xbest <- scale(dat[c(train, validation),c(predictors, response)])
Xtest <- scale(dat[test,c(predictors, response)])
x_test <- as.matrix(Xtest[, predictors])
y_test <- Xtest[,response]

ridge_best <- glmnet(Xbest[,predictors], Xbest[,response], alpha = 0, lambda = keep[4])
y_hat_test <- predict(ridge_fit, newx = x_test, s = keep[4])
mean((y_hat_test-y_test)^2)

Xall <- scale(dat)
ridge_model <- glmnet(Xall[,predictors], Xall[,response], alpha = 0, lambda = keep[4])
ridge_model$beta



