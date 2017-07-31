rm(list=ls())

train <- read.csv("./input/train.csv")
test <- read.csv("./input/test.csv")
sample_submission <- read.csv("./input/sample_submission.csv")

# Add the sales price variable to the test set 
test$SalePrice <- 0

# Combine the train and test data sets
combi <- rbind(train,test)

#################### Handling NA's #################################
str(train)
rev(sort(sapply(train, function(x) sum(is.na(x)))))

# Looking at the number of missing values and taking a closer look at the data description, 
# we can see that NA does not always mean that the values are missing. e.g.: In case of the feature Alley, 
# NA just means that there is "no alley access" to the house. Lets code such NA values to say "None".
# There are 2 types of variables in the data set, int and Factors
# for int, mark NAs as -1 and for factors, mark NAs as "Missing"

# Mark NA's in factors as "None"
for (col in colnames(combi)){
        if (class(combi[,col]) == "factor"){
                new_col = as.character(combi[,col])
                new_col[is.na(new_col)] <- "None"
                combi[col] = as.factor(new_col)
        }
}
# Fill remaining NA values (for numeric types) with -1
combi[is.na(combi)] = -1
# No NA's in the data now!!

################################################################################

#################FEATURE ENGINEERING############################################
# I noticed a couple key variables I was expecting to see were missing. Namely, 
# total square footage and total number of bathrooms are common features used to 
# classify homes, but these features are split up into different parts in the data set, 
# such as above grade square footage, basement square footage and so on. Let's add two new
# features for total square footage and total bathrooms:

# Drop Id
combi$Id <- NULL    

# Add variable that combines above grade living area with basement sq footage
combi$total_sq_footage = combi$GrLivArea + combi$TotalBsmtSF

# Add variable that combines above ground and basement full and half baths
combi$total_baths = combi$BsmtFullBath + combi$FullBath + (0.5 * (combi$BsmtHalfBath + combi$HalfBath))


# As we will use XGBoost as the predictive modeling technique, we will convert
# factor variables to numeric.

##########################Factors->Numeric#######################################

for (col in colnames(combi)) {
        if (class(combi[[col]])=="factor") {
                #cat("VARIABLE : ",col,"\n")
                levels <- levels(combi[[col]])
                combi[[col]] <- as.numeric(factor(combi[[col]], levels=levels))
        }
}

################################################################################
# Separate out the train and test sets
train <- combi[1:nrow(train),]
test <- combi[(nrow(train)+1):nrow(combi),]

# As there are 80 variables, let's find variables that are highly corelated to the SalePrice
library(reshape2)
cors = cor(train[ , sapply(train, is.numeric)])
cors_melted <- melt(cors)
cors_df1 <- cors_melted[cors_melted$Var1 == "SalePrice" & cors_melted$Var2 != "SalePrice" & abs(cors_melted$value)>0.5 ,]
cors_df1[order(-cors_df1$value),]
# Only 15 above 0.5 corelation level with OverallQuality being the highest.

# Let's look at the ones that are poorly coreleated to the SalePrice
cors_df2 <- cors_melted[cors_melted$Var1 == "SalePrice" & cors_melted$Var2 != "SalePrice" & abs(cors_melted$value)<0.2 ,]
cors_df2[order(-cors_df2$value),]

#The year and month sold don't appear to have much of a connection to sales prices.
#Interestingly, "overall condition" doesn't have a strong correlation to sales price, 
#while "overall quality" had the strongest correlation.

# Next, let's determine whether any of the numeric variables are highly correlated with one another.
# Though multicollinearity is not an issue in tree based models, we would still check.
# As there are so many variables, graphs will be difficult to decipher, thus, let's find out programatically
cors_df3 <- cors_melted[cors_melted$value>0.5 & cors_melted$value<1 & cors_melted$Var1 != "SalePrice" & cors_melted$Var2 != "SalePrice" ,]
cors_df3[order(-cors_df3$value),]

# The highest correlation is between GarageCars and GarageArea, which makes sense because we'd expect a 
# garage that can park more cars to have more area. Highly correlated variables can cause problems with certain types of 
# predictive models but since no variable pairs have a correlations above 0.9 and we will be using a tree-based model, let's keep them all.


# Now let's explore the distributions of the numeric variables with density plots. 
# This can help us get identify outlines and whether different variable and our target 
# variable are roughly normal, skewed or exhibit other oddities.
for (col in colnames(train)){
        if(is.numeric(train[,col])){
                plot(density(train[,col]), main=col)
        }
}

# A quick glance reveals that many of the numeric variables show right skew. Also,
# many variables have significant density near zero, indicating certain features are 
# only present in subset of homes. It also appears that far more homes sell in the spring 
# and summer months than winter. Lastly, the target variable SalePrice appears roughly normal, 
# but it has tail that goes off to the right, so a handful of homes sell for significantly more than the average. 
# Making accurate predictions for these pricey homes may be the most difficult part of making a good predictive model.

# Now we are ready to create a predictive model. Let's start by loading in some pacakges:
library(xgboost)
library(Metrics)

y_train <- train$SalePrice
train$SalePrice <- NULL
test$SalePrice <- NULL

x_train <- train
x_test <- test

dtrain = xgb.DMatrix(as.matrix(x_train), label=y_train)
dtest = xgb.DMatrix(as.matrix(x_test))

prediction <- as.data.frame(matrix(0,1459,20))
for(i in 1:20)
{
        xgb_params = list(
                seed = i,
                colsample_bytree = 0.7,
                subsample = 0.7,
                eta = 0.03,
                objective = 'reg:linear',
                max_depth = 6,
                num_parallel_tree = 1,
                min_child_weight = 1,
                base_score = 7
        )
        
        res = xgb.cv(xgb_params,
                     dtrain,
                     nrounds=500,
                     nfold=10,
                     early_stopping_rounds=20,
                     print_every_n = 10,
                     verbose= 1,
                     maximize=F)
        
        best_nrounds = res$best_iteration
        
        gbdt = xgb.train(xgb_params, dtrain, best_nrounds)
        
        prediction[,i] <- predict(gbdt,dtest)
}
sample_submission$SalePrice <- rowMeans(prediction)

#view variable importance plot
mat <- xgb.importance (feature_names = colnames(train),model = gbdt)
xgb.plot.importance (importance_matrix = mat[1:30]) 

write.csv(sample_submission, "HousePrices.csv", row.names = F)




