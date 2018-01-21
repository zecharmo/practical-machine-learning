### Practical Machine Learning Techniques Demonstration ###

# Context: A charity has a database of demographic and profile information on individuals
# to whom they have sent marketing communications in the past. This includes those who
# have donated and those who have not responded to the communications.

# Objectives: Identify which individuals should be contacted in their next mailing campaign
# and also predict how much they can expect in donations.

# Approach: This analysis will include a data exploration phase, development of a
# classification model to determine which indviduals should recieve mailers, and development
# of a regression model to predict the expected campaign profit. Each model development
# section will contain a demonstration of several machine learning techniques in an attempt
# to specify and score a top performing model.

library(MASS)
library(psych)
library(leaps)
library(bestglm)
library(glmnet)
library(pls)
library(splines)
library(tree)
library(randomForest)
library(gbm)
library(e1071)

# load data file
charity <- read.csv("C://Data//charity.csv",header=TRUE)

##### EXPLORATORY DATA ANALYSIS ######

# examine data structure - variable names and types
str(charity)

# ID Number = donor identification number, int
# REG1, REG2, REG3, REG4 = geographic region, binary
# HOME = whether donor is a homeowner or not, binary
# CHLD = number of children in household, int
# HINC = household income, factor (7 levels)
# GENF = donor sex, int
# WRAT = wealth rating, factor (9 levels)
# AVHV = average home value in donor's neighborhood, int
# INCM = median family income in donor's neighborhood, int
# INCA = average family income in donor's neighborhood, int
# PLOW = percent “low income” in donor's neighborhood, int
# NPRO = lifetime number of promotions received to date, int
# TGIF = dollar amount of lifetime gifts to date, int
# LGIF = dollar amount of largest gift to date, int
# RGIF = dollar amount of most recent gift, int
# TDON = number of months since last donation, int
# TLAG = number of months between first and second gift, int
# AGIF = average dollar amount of gifts to date, int
# DONR = classification response variable
# DAMT = prediction response variable

# 23 variables, 8,009 observations

# basic summary of each variable
summary(charity)

# basic distribution of each variable
describe(charity)

attach(charity)

# plots of variable distributions
# histograms of categorical data
hist(hinc)
hist(wrat)

# scatterplots of numerical data
plot(avhv)
plot(incm)
plot(inca)
plot(plow)
plot(npro)
plot(tgif)
plot(lgif)
plot(rgif)
plot(tdon)
plot(tlag)
plot(agif)
plot(damt)

# counts of binary indicator variables
sum(reg1) #1605
sum(reg2) #2555
sum(reg3) #1071
sum(reg4) #1117
sum(genf) #4848

# prepare data for correlation matrix - remove categoricals
c2=charity[6:21] #select only columns that aren't categorical
c2_d = subset(charity,donr==1) #select only known donor records
c2_nd = subset(charity,donr==0) #select only non-donor records
c2_d2=c2_d[6:21]
c2_nd2=c2_nd[6:21] 

# correlations
# A variable highly correlated with one (or both) or the dependent variables suggests that variable 
# may be a good predictor. Two independent variables highly correlated with each other suggests that 
# a model with only one of those variables would be sufficient as the second variable would only marginally 
# add to the predictive power
cor(c2)
cor(npro,tgif) #0.7058844
# this is the highest of all the correlations, there may be a relationship between the number of mailings and 
# the total amount donated, makes sense, but good news for the charity that continuing to contact results in donations

cor(c2_d2)
cor(c2_nd2)
# no easily observable profile differences between donors and non-donors


# the earlier plots reveal several variables with highly skewed distributions. 
# since many statistical learning techniques assume variables are normally distributed, 
# these highly skewed variables should be normalized by taking the log of each of these variables
# here we can apply variable transformations for highly skewed data
charity.t <- charity
charity.t$t_avhv <- log(charity.t$avhv)
charity.t$t_agif <- log(charity.t$agif)
charity.t$t_inca <- log(charity.t$inca)
charity.t$t_incm <- log(charity.t$incm)
charity.t$t_lgif <- log(charity.t$lgif)
charity.t$t_agif <- log(charity.t$agif)
charity.t$t_npro <- log(charity.t$npro)
charity.t$t_rgif <- log(charity.t$rgif)
charity.t$t_tdon <- log(charity.t$tdon)
charity.t$t_tgif <- log(charity.t$tgif)
charity.t$t_tlag <- log(charity.t$tlag)

# plots of variable distributions
# scatterplots of numerical data
plot(charity.t$t_avhv)
plot(charity.t$t_agif)
plot(charity.t$t_inca)
plot(charity.t$t_plow)
plot(charity.t$t_npro)
plot(charity.t$t_tgif)
plot(charity.t$t_lgif)
plot(charity.t$t_rgif)
plot(charity.t$t_tdon)
plot(charity.t$t_tlag)
plot(agif)
plot(damt)

# set up data for analysis
data.train <- charity.t[charity$part=="train",]
x.train <- data.train[,c(2:21,25:34)]
c.train <- data.train[,22] # donr
n.train.c <- length(c.train) # 3984
y.train <- data.train[c.train==1,23] # damt for observations with donr=1
n.train.y <- length(y.train) # 1995

data.valid <- charity.t[charity$part=="valid",]
x.valid <- data.valid[,c(2:21,25:34)]
c.valid <- data.valid[,22] # donr
n.valid.c <- length(c.valid) # 2018
y.valid <- data.valid[c.valid==1,23] # damt for observations with donr=1
n.valid.y <- length(y.valid) # 999

data.test <- charity.t[charity$part=="test",]
n.test <- dim(data.test)[1] # 2007
x.test <- data.test[,c(2:21,25:34)]


## just looking at training data

# counts of binary variables
sum(data.train$reg1) #816
sum(data.train$reg2) #1339
sum(data.train$reg3) #492
sum(data.train$reg4) #537
sum(data.train$genf) #2410

# histograms of categorical data
hist(data.train$hinc)
hist(data.train$wrat)

# scatterplots of transformed numerical data
plot(data.train$t_avhv)
plot(data.train$t_incm)
plot(data.train$t_inca)
plot(data.train$t_plow)
plot(data.train$t_npro)
plot(data.train$t_tgif)
plot(data.train$t_lgif)
plot(data.train$t_rgif)
plot(data.train$t_tdon)
plot(data.train$t_tlag)
plot(data.train$t_agif)
plot(data.train$t_damt)

# are there different patterns between donors and non donors?
# cross tabs of income groups by donor/non-donor
tab1 = table(data.train$hinc,data.train$donr)
tab1
prop.table(tab1,2)

#   0    1
#1  183   29
#2  322  145
#3  202  205
#4  609 1226 <<more donors at middle income group
#5  309  274
#6  190   70
#7  174   46

tab2 = table(data.train$genf,data.train$donr)
tab2
prop.table(tab2,2)

#    0    1
#0  769  805
#1 1220 1190

tab3 = table(data.train$npro,data.train$donr)
tab3
prop.table(tab3,2)

tab4 = table(data.train$wrat,data.train$donr)
tab4
prop.table(tab4,2)

# correlations
cor(data.train$wrat,data.train$donr)
# 0.2492664, being a donor not strongly correlated with wealth rating

cor(data.train$hinc,data.train$donr)
# 0.02772893, being a donor not correlated with income

cor(data.train$npro,data.train$donr)
# 0.135723, being a donor not correlated with number of mailings

##### DATA PREPARATION ######

# standardize data set
x.train.mean <- apply(x.train, 2, mean)
x.train.sd <- apply(x.train, 2, sd)
x.train.std <- t((t(x.train)-x.train.mean)/x.train.sd) # standardize to have zero mean and unit standard deviation
apply(x.train.std, 2, mean) # check zero mean
apply(x.train.std, 2, sd) # check unit standard deviation
data.train.std.c <- data.frame(x.train.std, donr=c.train) # to classify donr
data.train.std.y <- data.frame(x.train.std[c.train==1,], damt=y.train) # to predict damt when donr=1

x.valid.std <- t((t(x.valid)-x.train.mean)/x.train.sd) # standardize to have zero mean and unit standard deviation
data.valid.std.c <- data.frame(x.valid.std, donr=c.valid) # to classify donr
data.valid.std.y <- data.frame(x.valid.std[c.valid==1,], damt=y.valid) # to predict damt when donr=1

x.test.std <- t((t(x.test)-x.train.mean)/x.train.sd) # standardize to have zero mean and unit standard deviation
data.test.std <- data.frame(x.test.std)

# observe differences in distributions from original data set
summary(data.train.std.c$hinc)

hist(data.train.std.c$hinc)
hist(data.train.std.c$wrat)

plot(data.train.std.c$t_avhv)
plot(data.train.std.c$t_incm)
plot(data.train.std.c$t_inca)
plot(data.train.std.c$t_plow)
plot(data.train.std.c$t_npro)
plot(data.train.std.c$t_tgif)
plot(data.train.std.c$t_lgif)
plot(data.train.std.c$t_rgif)
plot(data.train.std.c$t_tdon)
plot(data.train.std.c$t_tlag)
plot(data.train.std.c$t_agif)
plot(data.train.std.y$t_damt)

sum(data.train.std.c$reg1)
sum(data.train.std.c$reg2)
sum(data.train.std.c$reg3)
sum(data.train.std.c$reg4)
sum(data.train.std.c$genf)

c2=data.train.std.c
c2_d = subset(data.train.std.c,donr==1,select=c(-t_plow))
c2_nd = subset(data.train.std.c,donr==0,select=c(-t_plow))

cor(c2_d)

##### CLASSIFICATION MODELING ######

## Linear Discriminant Analysis
# Linear discriminant analysis describes the categorical dependent variable as a linear combination 
# of the continuous, normally distributed predictor variables. It uses the conditional probability 
# density function and Bayes’ Theorem to assign a case to a class if the log likelihood ratio falls 
# below a set threshold.

# full model - all original variables
model.lda1 <- lda(donr ~ reg1 + reg2 + reg3 + reg4 + home + chld + hinc + genf + wrat + avhv + incm + inca + 
			plow + npro + tgif + lgif + rgif + tdon + tlag + agif, data.train.std.c)

# full model - all transformed variables
model.lda2 <- lda(donr ~ reg1 + reg2 + reg3 + reg4 + home + chld + hinc + genf + wrat + t_avhv + t_incm + 
			t_inca + plow + t_npro + t_tgif + t_lgif + t_rgif + t_tdon + tlag + t_agif, data.train.std.c) 

# model with logical predictors
model.lda3 <- lda(donr ~ home + chld + hinc + genf + wrat + npro + tgif + rgif + tdon + tlag, data.train.std.c)

# model with logical predictors (transformed)
model.lda4 <- lda(donr ~ home + chld + hinc + genf + wrat + t_npro + t_tgif + t_rgif + t_tdon + tlag, data.train.std.c)

# model with significant predictors exponentiated
model.lda5 <- lda(donr ~ reg1 + reg2 + reg3 + reg4 + home + chld + hinc + I(hinc^2) + genf + wrat + I(wrat^2) + t_avhv + 
			t_incm + t_inca + plow + t_npro + t_tgif + t_lgif + t_rgif + t_tdon + tlag + t_agif, data.train.std.c) 

# model with t_npro also exponentiated and plow removed
model.lda6 <- lda(donr ~ reg1 + reg2 + reg3 + reg4 + home + chld + hinc + I(hinc^2) + genf + wrat + I(wrat^2) + t_avhv + 
			t_incm + t_inca + t_npro + I(t_npro^2) + t_tgif + t_lgif + t_rgif + t_tdon + tlag + t_agif, data.train.std.c)

# model 6 with plow added in
model.lda7 <- lda(donr ~ reg1 + reg2 + reg3 + reg4 + home + chld + hinc + I(hinc^2) + genf + wrat + I(wrat^2) + t_avhv + t_incm + 
			t_inca + plow + t_npro + I(t_npro^2) + t_tgif + t_lgif  + t_rgif  + t_tdon + tlag + t_agif, data.train.std.c)

# model 7 with t_npro removed
model.lda8 <- lda(donr ~ reg1 + reg2 + reg3 + reg4 + home + chld + hinc + I(hinc^2) + genf + wrat + I(wrat^2) + avhv + incm + 
			inca + plow + npro + tgif + lgif + rgif + tdon + tlag + agif, data.train.std.c)
 
model.lda8 <- lda(donr ~ reg1 + reg2 + reg3 + reg4 + home + chld + hinc + I(hinc^2) + genf + wrat + I(wrat^2) + avhv + 
			incm + inca + plow + npro + tgif + lgif + rgif + tdon + tlag + agif, data.train.std.c)

# calculate predictions on validation set

post.valid.lda1 <- predict(model.lda1, data.valid.std.c)$posterior[,2] # n.valid.c post probs
post.valid.lda2 <- predict(model.lda2, data.valid.std.c)$posterior[,2]
post.valid.lda3 <- predict(model.lda3, data.valid.std.c)$posterior[,2]
post.valid.lda4 <- predict(model.lda4, data.valid.std.c)$posterior[,2]
post.valid.lda5 <- predict(model.lda5, data.valid.std.c)$posterior[,2]
post.valid.lda6 <- predict(model.lda6, data.valid.std.c)$posterior[,2]
post.valid.lda7 <- predict(model.lda7, data.valid.std.c)$posterior[,2]
post.valid.lda8 <- predict(model.lda8, data.valid.std.c)$posterior[,2]

# ordered profit calculations

profit.lda1 <- cumsum(14.5*c.valid[order(post.valid.lda1, decreasing=T)]-2)
plot(profit.lda1) # see how profits change as more mailings are made
n.mail.valid <- which.max(profit.lda1) # number of mailings that maximizes profits
c(n.mail.valid, max(profit.lda1)) # report number of mailings and maximum profit
# 1472.0 11367.5

cutoff.lda1 <- sort(post.valid.lda1, decreasing=T)[n.mail.valid+1] # set cutoff based on n.mail.valid
chat.valid.lda1 <- ifelse(post.valid.lda1>cutoff.lda1, 1, 0) # mail to everyone above the cutoff
table(chat.valid.lda1, c.valid) # classification table
#               c.valid
#chat.valid.lda1   0   1
#              0 534  12
#              1 485 987

profit.lda2 <- cumsum(14.5*c.valid[order(post.valid.lda2, decreasing=T)]-2)
plot(profit.lda2) 
n.mail.valid <- which.max(profit.lda2) 
c(n.mail.valid, max(profit.lda2)) 
#1392.0 11353.5

cutoff.lda2 <- sort(post.valid.lda2, decreasing=T)[n.mail.valid+1] 
chat.valid.lda2 <- ifelse(post.valid.lda2>cutoff.lda2, 1, 0) 
table(chat.valid.lda2, c.valid) 
#chat.valid.lda2   0   1
#              0 602  24
#              1 417 975

profit.lda3 <- cumsum(14.5*c.valid[order(post.valid.lda3, decreasing=T)]-2)
plot(profit.lda3) 
n.mail.valid <- which.max(profit.lda3) 
c(n.mail.valid, max(profit.lda3)) 
#1592 11055

cutoff.lda3 <- sort(post.valid.lda3, decreasing=T)[n.mail.valid+1] 
chat.valid.lda3 <- ifelse(post.valid.lda3>cutoff.lda3, 1, 0) 
table(chat.valid.lda3, c.valid)
#               c.valid
#chat.valid.lda3   0   1
#              0 409  17
#              1 610 982

profit.lda4 <- cumsum(14.5*c.valid[order(post.valid.lda4, decreasing=T)]-2)
plot(profit.lda4) 
n.mail.valid <- which.max(profit.lda4) 
c(n.mail.valid, max(profit.lda4)) 
#1598 11072

cutoff.lda4 <- sort(post.valid.lda4, decreasing=T)[n.mail.valid+1] 
chat.valid.lda4 <- ifelse(post.valid.lda4>cutoff.lda4, 1, 0) 
table(chat.valid.lda4, c.valid)
#               c.valid
#chat.valid.lda4   0   1
#              0 405  15
#              1 614 984

profit.lda5 <- cumsum(14.5*c.valid[order(post.valid.lda5, decreasing=T)]-2) 
n.mail.valid <- which.max(profit.lda5) 
c(n.mail.valid, max(profit.lda5)) 
#1383 11647
plot(profit.lda5)

cutoff.lda5 <- sort(post.valid.lda5, decreasing=T)[n.mail.valid+1] 
chat.valid.lda5 <- ifelse(post.valid.lda5>cutoff.lda5, 1, 0) 
table(chat.valid.lda5, c.valid)
#               c.valid
#chat.valid.lda5   0   1
#              0 630   5
#              1 389 994

profit.lda6 <- cumsum(14.5*c.valid[order(post.valid.lda6, decreasing=T)]-2) 
n.mail.valid <- which.max(profit.lda6) 
c(n.mail.valid, max(profit.lda6)) 
#1373.0 11652.5
plot(profit.lda6)

cutoff.lda6 <- sort(post.valid.lda6, decreasing=T)[n.mail.valid+1] 
chat.valid.lda6 <- ifelse(post.valid.lda6>cutoff.lda6, 1, 0) 
table(chat.valid.lda6, c.valid)
#               c.valid
#chat.valid.lda6   0   1
#              0 639   6
#              1 380 993

profit.lda7 <- cumsum(14.5*c.valid[order(post.valid.lda7, decreasing=T)]-2) 
n.mail.valid <- which.max(profit.lda7) 
c(n.mail.valid, max(profit.lda7)) 
#1374.0 11650.5
plot(profit.lda7)

cutoff.lda7 <- sort(post.valid.lda7, decreasing=T)[n.mail.valid+1] 
chat.valid.lda7 <- ifelse(post.valid.lda7>cutoff.lda7, 1, 0) 
table(chat.valid.lda7, c.valid)
#               c.valid
#chat.valid.lda7   0   1
#              0 638   6
#              1 381 993

profit.lda8 <- cumsum(14.5*c.valid[order(post.valid.lda8, decreasing=T)]-2) 
n.mail.valid <- which.max(profit.lda8) 
c(n.mail.valid, max(profit.lda8)) 
#1328.0 11655.5
plot(profit.lda8)

cutoff.lda8 <- sort(post.valid.lda8, decreasing=T)[n.mail.valid+1] 
chat.valid.lda8 <- ifelse(post.valid.lda8>cutoff.lda8, 1, 0) 
table(chat.valid.lda8, c.valid)
#               c.valid
#chat.valid.lda8   0   1
#              0 678  12
#              1 341 987

# model 8 has the highest profit 11655.5 with 1328 mailings

## Logistic Regression ##
# Logistic regression models assign cases to classes by predicting the probability that a case belongs 
# to that class. Probabilities are estimated using a logistic function, a special case of a generalized 
# linear model, where a logit link function converts the linear regression function into a probability by 
# ensuring the model output is restricted to the range (0,1)

# best model from LDA
model.log1 <- glm(donr ~ reg1 + reg2 + reg3 + reg4 + home + chld + hinc + I(hinc^2) + genf + wrat + 
                    avhv + incm + inca + plow + npro + tgif + lgif + rgif + tdon + tlag + agif, 
                  data.train.std.c, family=binomial("logit"))

# model with significant predictors exponentiated
model.log2 <- glm(donr ~ reg1 + reg2 + reg3 + reg4 + home + chld + hinc + I(hinc^2) + genf + wrat + I(wrat^2) +
                    t_avhv + t_incm + t_inca + plow + t_npro + I(t_npro^2) + t_tgif + t_lgif  + t_rgif  + t_tdon + tlag + t_agif,
                   data.train.std.c, family=binomial("logit"))

# model 2 without t_npro exponentiated
model.log3 <- glm(donr ~ reg1 + reg2 + reg3 + reg4 + home + chld + hinc + I(hinc^2) + genf + wrat + I(wrat^2) +
                    t_avhv + t_incm + t_inca + plow + t_npro + t_tgif + t_lgif  + t_rgif  + t_tdon + tlag + t_agif,
                   data.train.std.c, family=binomial("logit"))

# model 3 with non-transformed variables
model.log4 <- glm(donr ~ reg1 + reg2 + reg3 + reg4 + home + chld + hinc + I(hinc^2) + genf + wrat + I(wrat^2) +
                    avhv + incm + inca + plow + npro + tgif + lgif  + rgif  + tdon + tlag + agif,
                   data.train.std.c, family=binomial("logit"))

# model 3 without plow
model.log5 <- glm(donr ~ reg1 + reg2 + reg3 + reg4 + home + chld + hinc + I(hinc^2) + genf + wrat + I(wrat^2) +
                    avhv + incm + inca + npro + tgif + lgif  + rgif  + tdon + tlag + agif,
                   data.train.std.c, family=binomial("logit"))

# model 3 with plow exponentiated
model.log6 <- glm(donr ~ reg1 + reg2 + reg3 + reg4 + home + chld + hinc + I(hinc^2) + genf + wrat + I(wrat^2) +
                    avhv + incm + inca + plow + I(plow^2) + npro + tgif + lgif  + rgif  + tdon + tlag + agif,
                   data.train.std.c, family=binomial("logit"))

# calculate predictions

post.valid.log1 <- predict(model.log1, data.valid.std.c, type="response")
post.valid.log2 <- predict(model.log2, data.valid.std.c, type="response")
post.valid.log3 <- predict(model.log3, data.valid.std.c, type="response") 
post.valid.log4 <- predict(model.log4, data.valid.std.c, type="response") 
post.valid.log5 <- predict(model.log5, data.valid.std.c, type="response") 
post.valid.log6 <- predict(model.log6, data.valid.std.c, type="response") 

# calculate ordered profit 

profit.log1 <- cumsum(14.5*c.valid[order(post.valid.log1, decreasing=T)]-2)
plot(profit.log1) # see how profits change as more mailings are made
n.mail.valid <- which.max(profit.log1) # number of mailings that maximizes profits
c(n.mail.valid, max(profit.log1)) # report number of mailings and maximum profit
# 1321.0 11640.5

cutoff.log1 <- sort(post.valid.log1, decreasing=T)[n.mail.valid+1] # set cutoff based on n.mail.valid
chat.valid.log1 <- ifelse(post.valid.log1>cutoff.log1, 1, 0) # mail to everyone above the cutoff
table(chat.valid.log1, c.valid) # classification table
#               c.valid
#chat.valid.log1   0   1
#              0 683  14
#              1 336 985

profit.log2 <- cumsum(14.5*c.valid[order(post.valid.log2, decreasing=T)]-2)
n.mail.valid <- which.max(profit.log2) 
c(n.mail.valid, max(profit.log2)) 
#1334 11629
plot(profit.log2)

cutoff.log2 <- sort(post.valid.log2, decreasing=T)[n.mail.valid+1]
chat.valid.log2 <- ifelse(post.valid.log2>cutoff.log2, 1, 0)
table(chat.valid.log2, c.valid)
#               c.valid
#chat.valid.log2   0   1
#              0 671  13
#              1 348 986

profit.log3 <- cumsum(14.5*c.valid[order(post.valid.log3, decreasing=T)]-2)
n.mail.valid <- which.max(profit.log3) 
c(n.mail.valid, max(profit.log3)) 
#1325.0 11632.5
plot(profit.log3)

cutoff.log3 <- sort(post.valid.log3, decreasing=T)[n.mail.valid+1]
chat.valid.log3 <- ifelse(post.valid.log3>cutoff.log3, 1, 0)
table(chat.valid.log3, c.valid)

profit.log4 <- cumsum(14.5*c.valid[order(post.valid.log4, decreasing=T)]-2)
n.mail.valid <- which.max(profit.log4) 
c(n.mail.valid, max(profit.log4)) 
#1260 11661
plot(profit.log4)

cutoff.log4 <- sort(post.valid.log4, decreasing=T)[n.mail.valid+1]
chat.valid.log4 <- ifelse(post.valid.log4>cutoff.log4, 1, 0)
table(chat.valid.log4, c.valid)
#               c.valid
#chat.valid.log4   0   1
#              0 737  21
#              1 282 978

profit.log5 <- cumsum(14.5*c.valid[order(post.valid.log5, decreasing=T)]-2)
n.mail.valid <- which.max(profit.log5) 
c(n.mail.valid, max(profit.log5)) 
#1218 11629
plot(profit.log5)

cutoff.log5 <- sort(post.valid.log5, decreasing=T)[n.mail.valid+1]
chat.valid.log5 <- ifelse(post.valid.log5>cutoff.log5, 1, 0)
table(chat.valid.log5, c.valid)
#               c.valid
#chat.valid.log5   0   1
#              0 771  29
#              1 248 970

profit.log6 <- cumsum(14.5*c.valid[order(post.valid.log6, decreasing=T)]-2)
n.mail.valid <- which.max(profit.log6) 
c(n.mail.valid, max(profit.log6)) 
#1271 11639
plot(profit.log5)

cutoff.log6 <- sort(post.valid.log6, decreasing=T)[n.mail.valid+1]
chat.valid.log6 <- ifelse(post.valid.log6>cutoff.log6, 1, 0)
table(chat.valid.log6, c.valid)

log.1 = bestglm(data.train.std.c, IC = "BIC", TopModels = 5)

# model recommended by best glm procedure - best subset variable reduction for glms
model.log7 <- glm(donr ~ reg1 + reg2 + home + chld + wrat + tdon + tlag + t_incm + t_tdon + t_tgif,
                   data.train.std.c, family=binomial("logit"))

post.valid.log7 <- predict(model.log7, data.valid.std.c, type="response") # n.valid post probs

profit.log7 <- cumsum(14.5*c.valid[order(post.valid.log7, decreasing=T)]-2)
n.mail.valid <- which.max(profit.log7) 
c(n.mail.valid, max(profit.log7)) 
#1354.0 11400.5
plot(profit.log7)

cutoff.log7 <- sort(post.valid.log7, decreasing=T)[n.mail.valid+1]
chat.valid.log7 <- ifelse(post.valid.log7>cutoff.log7, 1, 0)
table(chat.valid.log7, c.valid)
#               c.valid
#chat.valid.log7   0   1
#              0 638  26
#              1 381 973

#best glm model with squared predictors from highest profit model above
model.log8 <- glm(donr ~ reg1 + reg2 + home + chld + hinc + I(hinc^2) + wrat + I(wrat^2) + tdon + tlag + t_incm + t_tdon + t_tgif,
                   data.train.std.c, family=binomial("logit"))

post.valid.log8 <- predict(model.log8, data.valid.std.c, type="response") # n.valid post probs

profit.log8 <- cumsum(14.5*c.valid[order(post.valid.log8, decreasing=T)]-2)
n.mail.valid <- which.max(profit.log8) 
c(n.mail.valid, max(profit.log8)) #1322.0 11667.5
plot(profit.log8)

cutoff.log8 <- sort(post.valid.log8, decreasing=T)[n.mail.valid+1]
chat.valid.log8 <- ifelse(post.valid.log8>cutoff.log8, 1, 0)
table(chat.valid.log8, c.valid)

               c.valid
chat.valid.log8   0   1
              0 684  12
              1 335 987

# model recommended by best fitting gam regression model
model.log9 <- glm(donr ~ reg1 + reg2 + home + chld + poly(hinc,2,raw=T) + poly(wrat,2,raw=T) + poly(plow,3,raw=T) + tdon + tlag + t_incm + t_tdon + t_tgif,
                   data.train.std.c, family=binomial("logit"))

post.valid.log9 <- predict(model.log9, data.valid.std.c, type="response") # n.valid post probs

profit.log9 <- cumsum(14.5*c.valid[order(post.valid.log9, decreasing=T)]-2)
n.mail.valid <- which.max(profit.log9) 
c(n.mail.valid, max(profit.log9)) #1315 11667

cutoff.log9 <- sort(post.valid.log9, decreasing=T)[n.mail.valid+1]
chat.valid.log9 <- ifelse(post.valid.log9>cutoff.log9, 1, 0)
table(chat.valid.log9, c.valid)
               c.valid
chat.valid.log9   0   1
              0 690  13
              1 329 986

# Highest profit models are model 8 and 9 (11667.5 & 11667) with 1322 and 1315 mailings

## Decision Trees and Random Forests ##
# A decision tree splits observations into regions based on their value for a series of variables, 
# or decision nodes, where the series of splits follows the shape of an inverted tree. An observation 
# is classified into a group based on the most commonly occurring class of the observations in that region

# simple classification tree model with all predictors
model.tree1 <- tree(factor(donr) ~.,data=data.train.std.c)
summary(model.tree1)
# "chld" "home" "wrat" "reg2" "hinc" "tlag" "tdon"
# Residual mean deviance:  0.6833 = 2711 / 3968
# Misclassification error rate: 0.1426 = 568 / 3984
plot(model.tree1) + text(model.tree1)

tree.pred1=predict(model.tree1,data.valid.std.c,type="class")

profit.tree1 <- cumsum(14.5*c.valid[order(tree.pred1, decreasing=T)]-2)
n.mail.valid <- which.max(profit.tree1) 
c(n.mail.valid, max(profit.tree1)) #1168 11149

table(tree.pred1,c.valid)
          c.valid
tree.pred1   0   1
         0 783  70
         1 236 929

set.seed(3)
cv.tree1 =cv.tree(model.tree1,FUN=prune.misclass) #prune tree to improve classification

par(mfrow=c(1,2)) +
plot(cv.tree1$size ,cv.tree1$dev ,type="b") +
plot(cv.tree1$k ,cv.tree1$dev,type="b")
# original tree had 16 terminal nodes, CV reveals error is lowest with about 15
# not much to be gained by pruning this tree

# random forest bagging model to reduce variance
set.seed(1)
model.bag1= randomForest(factor(donr)~.,data=data.train.std.c,mtry=30,importance =TRUE)
# OOB error 11.57%
     0    1 class.error
0 1761  228   0.1146305
1  233 1762   0.1167920
importance(model.bag1)
varImpPlot(model.bag1)
#indicates most important variables are chld,home,hinc,reg2,wrat,t_tdon,plow,incm,t_tgif

bag.pred1=predict(model.bag1,newdata=data.valid.std.c,type="class")

profit.bag1 <- cumsum(14.5*c.valid[order(bag.pred1, decreasing=T)]-2)
n.mail.valid <- which.max(profit.bag1) 
c(n.mail.valid, max(profit.bag1)) #1033 11071

table(bag.pred1,c.valid)
         c.valid
bag.pred1   0   1
        0 892  93
        1 127 906

# reduce number of variables tried at each split by 1/3

set.seed(1)
model.bag2= randomForest(factor(donr)~.,data=data.train.std.c,mtry=10,importance =TRUE)
model.bag2
# OOB error 11.04%
     0    1 class.error
0 1761  228   0.1146305
1  212 1783   0.1062657
importance(model.bag2)
varImpPlot(model.bag2)
#indicates most important variables are chld,home,hinc,reg2,t_tdon,t_incm,t_tgif

bag.pred2=predict(model.bag2,newdata=data.valid.std.c,type="class")

profit.bag2 <- cumsum(14.5*c.valid[order(bag.pred2, decreasing=T)]-2)
n.mail.valid <- which.max(profit.bag2) 
c(n.mail.valid, max(profit.bag2)) #1057 11139

table(bag.pred2,c.valid)
         c.valid
bag.pred2   0   1
        0 879  86
        1 140 913

# try reducing the number of predictors 

set.seed(1)
model.bag3= randomForest(factor(donr)~.,data=data.train.std.c,mtry=7,importance =TRUE)
model.bag3
# OOB error 11.07%
     0    1 class.error
0 1750  239   0.1201609
1  202 1793   0.1012531
importance(model.bag3)
varImpPlot(model.bag3)
#indicates most important variables are chld,home,hinc,reg2,t_tdon,t_incm,t_tgif

bag.pred3=predict(model.bag3,newdata=data.valid.std.c,type="class")

profit.bag3 <- cumsum(14.5*c.valid[order(bag.pred3, decreasing=T)]-2)
n.mail.valid <- which.max(profit.bag3) 
c(n.mail.valid, max(profit.bag3)) #1060.0 11176.5

#table(bag.pred3,c.valid)
#         c.valid
#bag.pred3   0   1
#        0 879  83
#        1 140 916


# boosting models
# the boosting procedure constructs trees sequentially rather than in tandem, so that successive trees 
# are informed by the prior trees. Where single trees and trees built in tandem are “greedy” and make 
# the best split at each node, trees produced sequentially learn from previous trees and produce 
# “better” splits at each node improving performance
set.seed(1)
model.boost1=gbm(donr~.,data=data.train.std.c, distribution="bernoulli",n.trees=5000, interaction.depth=5)
summary(model.boost1)
# most influential: chld, hinc, reg2, home, wrat, tdon, incm, tgif, tlag, reg1

boost.pred1=predict(model.boost1,newdata=data.valid.std.c,type="response",n.trees=5000)

profit.boost1 <- cumsum(14.5*c.valid[order(boost.pred1, decreasing=T)]-2)
n.mail.valid <- which.max(profit.boost1) 
c(n.mail.valid, max(profit.boost1)) #1216.0 11850.5


model.boost2=gbm(donr~.,data=data.train.std.c, distribution="bernoulli",n.trees=5000, interaction.depth=5,shrinkage =0.2,verbose=F)
summary(model.boost2)
# most influential: chld, agif, avhv,tgif, hinc, npro, incm, wrat, inca, tdon, reg2, plow, lgif

boost.pred2=predict(model.boost2,newdata=data.valid.std.c,type="response",n.trees=5000)

profit.boost2 <- cumsum(14.5*c.valid[order(boost.pred2, decreasing=T)]-2)
n.mail.valid <- which.max(profit.boost2) 
c(n.mail.valid, max(profit.boost2)) #1261 11775


set.seed(1)
model.boost3=gbm(donr~.,data=data.train.std.c, distribution="bernoulli",n.trees=5000, interaction.depth=10)
summary(model.boost3)
# most influential: chld, hinc, reg2, home, wrat, tdon, incm, tgif, tlag, reg1

boost.pred3=predict(model.boost3,newdata=data.valid.std.c,type="response",n.trees=5000)

profit.boost3 <- cumsum(14.5*c.valid[order(boost.pred3, decreasing=T)]-2)
n.mail.valid <- which.max(profit.boost3) 
c(n.mail.valid, max(profit.boost3)) #1242 11871

table(boost.pred3,c.valid)

# best decision tree model produces 11871 in profit with 1242 mailings, this is the best model overall


## Support Vector Machines ##

# begin with a linear kernel
set.seed(1)
tune.out=tune(svm,factor(donr)~., data=data.train.std.c,kernel ="linear",ranges=list(cost=c(0.001, 0.01, 0.1, 1,5,10,100) ))
# best parameter is cost of 5
# best performance is 0.1510976

model.svm1=tune.out$best.model
summary(model.svm1) #1432 support vectors

svm.pred1=predict(model.svm1,newdata=data.valid.std.c)

profit.svm1 <- cumsum(14.5*c.valid[order(svm.pred1, decreasing=T)]-2)
n.mail.valid <- which.max(profit.svm1) 
c(n.mail.valid, max(profit.svm1)) #1077.0 10591.5

#table(svm.pred1,c.valid)
#         c.valid
#svm.pred1   0   1
#        0 830 122
#        1 189 877

# polynomial kernel
set.seed(1)
tune.out2=tune(svm,factor(donr)~., data=data.train.std.c,kernel ="polynomial",ranges=list(cost=c(0.001, 0.01, 0.1, 1,5,10,100) ))
summary(tune.out2)
# best parameter is cost of 1
# best performance is 0.1633921

model.svm2=tune.out2$best.model
summary(model.svm2) #2088 support vectors

svm.pred2=predict(model.svm2,newdata=data.valid.std.c)

profit.svm2 <- cumsum(14.5*c.valid[order(svm.pred2, decreasing=T)]-2)
n.mail.valid <- which.max(profit.svm2) 
c(n.mail.valid, max(profit.svm2)) #1145 10731

table(svm.pred2,c.valid)
#         c.valid
#svm.pred2   0   1
#        0 772 101
#        1 247 898

# radial kernel
set.seed(1)
tune.out3=tune(svm,factor(donr)~., data=data.train.std.c,kernel ="radial",ranges=list(cost=c(0.001, 0.01, 0.1, 1,5,10,100) ))
summary(tune.out3)
# best parameter is cost of 5
# best performance is 0.1207277

model.svm3=tune.out3$best.model
summary(model.svm3) #1466 support vectors

svm.pred3=predict(model.svm3,newdata=data.valid.std.c)

profit.svm3 <- cumsum(14.5*c.valid[order(svm.pred3, decreasing=T)]-2)
n.mail.valid <- which.max(profit.svm3) 
c(n.mail.valid, max(profit.svm3)) #1068.0 11015.5

table(svm.pred3,c.valid)
#         c.valid
#svm.pred3   0   1
#        0 858  92
#        1 161 907

##### PREDICTION MODELING ######

## Least squares regression ##
# Linear regression is a simple yet powerful method that assumes a linear relationship between 
# the response (gift amount) and predictor variables where a model is fit such that it minimizes 
# the sum of squared errors between each data point and the fitted line

model.ls1 <- lm(damt ~ reg1 + reg2 + reg3 + reg4 + home + chld + hinc + genf + wrat + 
                  avhv + incm + inca + plow + npro + tgif + lgif + rgif + tdon + tlag + agif, 
                data.train.std.y) #full model with original variables

pred.valid.ls1 <- predict(model.ls1, newdata = data.valid.std.y) # validation predictions
mean((y.valid - pred.valid.ls1)^2) # mean prediction error
# 1.866657
sd((y.valid - pred.valid.ls1)^2)/sqrt(n.valid.y) # std error
# 0.1695524

model.ls2 <- lm(damt ~ reg1 + reg2 + reg3 + reg4 + home + chld + hinc + genf + wrat + 
                  t_avhv + t_incm + t_inca + plow + t_npro + t_tgif + t_lgif + t_rgif + t_tdon + t_tlag + t_agif, 
                data.train.std.y) #full model with transformed variables

pred.valid.ls2 <- predict(model.ls2, newdata = data.valid.std.y) # validation predictions
mean((y.valid - pred.valid.ls2)^2) # mean prediction error
# 1.556639
sd((y.valid - pred.valid.ls2)^2)/sqrt(n.valid.y) # std error
# 0.1608402

#perform subset selection to select only important variables

model.sub <- regsubsets(damt ~.,data=data.train.std.y,nvmax=30)
summary.sub = summary(model.sub)
summary.sub$adjr2
coef(model.sub,22) # to get the predictor variables
# the model with the highest adjusted r2 has 22 variables

model.sub2 <- regsubsets(damt ~.,data=data.train.std.y,nvmax=30,method="forward")
summary.sub2 = summary(model.sub2)
summary.sub2$adjr2
# best adjr2 is model with 21 predictors, but r2 is less than "best" subset selection

model.sub3 <- regsubsets(damt ~.,data=data.train.std.y,nvmax=30,method="backward")
summary.sub3 = summary(model.sub3)
summary.sub3$adjr2
# best adjr2 is again the model with 22 predictors, the same as "best" subset selection
coef(model.sub3,22) # to get the predictor variables, the same model as best subset

model.ls3 <- lm(damt ~ reg2 + reg3 + reg4 + home + chld + hinc + genf + inca + plow + npro + tgif + lgif +
                  rgif + tdon + t_incm + t_inca + t_agif + t_tgif + t_lgif + t_rgif + t_tdon + t_tlag , 
                data.train.std.y)

pred.valid.ls3 <- predict(model.ls3, newdata = data.valid.std.y) # validation predictions
mean((y.valid - pred.valid.ls3)^2) # mean prediction error
# 1.539
sd((y.valid - pred.valid.ls3)^2)/sqrt(n.valid.y) # std error
# 0.1608116

#some of these predictors are not significant, does the model fit better if we remove them?

model.ls4 <- lm(damt ~ reg2 + reg3 + reg4 + home + chld + hinc + genf + inca + plow + lgif +
                  tdon + t_incm + t_inca + t_agif + t_tgif + t_lgif + t_rgif, 
                data.train.std.y)

pred.valid.ls4 <- predict(model.ls4, newdata = data.valid.std.y) # validation predictions
mean((y.valid - pred.valid.ls4)^2) # mean prediction error
# 1.547276 - mpe increases
sd((y.valid - pred.valid.ls4)^2)/sqrt(n.valid.y) # std error
# 0.1607696 - std error decreases slightly

# best model is model 3 with MSE of 1.539 and SD of 0.161


## Ridge regression ##
# Shrinkage methods attempt to regularize the coefficient estimates by shrinking them towards zero. 
# Ridge regression uses a tuning parameter to shrink all coefficients towards zero 

# set up model matrix with predictors
x=model.matrix(damt~.,data.train.std.y)[,-1]
y=data.train.std.y$damt

# set lambda grid
grid=10^seq(10,-2, length =100)

model.rdg1=glmnet(x,y,alpha=0, lambda=grid)

set.seed(1)
cv.out=cv.glmnet(x,y,alpha=0)
plot(cv.out)
bestlam =cv.out$lambda.min
bestlam #0.1252296

model.rdg2=glmnet(x,y,alpha=0, lambda=0.1252296)

newx=model.matrix(damt~.,data.valid.std.y)[,-1]
y.test=data.valid.std.y$damt

ridge.pred=predict(model.rdg2,s=bestlam,newx=newx)
mean((ridge.pred -y.test)^2)
#1.567331


x1=model.matrix(damt~ reg1 + reg2 + reg3 + reg4 + home + chld + hinc + genf + wrat + 
                  t_avhv + t_incm + t_inca + plow + t_npro + t_tgif + t_lgif + t_rgif + 
                   t_tdon + t_tlag + t_agif,data.train.std.y)[,-1]


model.rdg3=glmnet(x1,y,alpha=0, lambda=grid)

set.seed(1)
cv.out=cv.glmnet(x1,y,alpha=0)
plot(cv.out)
bestlam =cv.out$lambda.min
bestlam #0.1252296

model.rdg4=glmnet(x1,y,alpha=0, lambda=0.1252296)

newx=model.matrix(damt~reg1 + reg2 + reg3 + reg4 + home + chld + hinc + genf + wrat + 
                  t_avhv + t_incm + t_inca + plow + t_npro + t_tgif + t_lgif + t_rgif + 
                   t_tdon + t_tlag + t_agif,data.valid.std.y)[,-1]

ridge.pred=predict(model.rdg4,s=bestlam,newx=newx)
mean((ridge.pred -y.test)^2)
#1.572448

# best model is model 2 with MSE of 1.567


## Lasso ##
# The lasso is another shrinkage method that uses a different penalty term to shrink some coefficients 
# all the way to zero thus performing variable reduction as well.

model.lasso1=glmnet(x,y,alpha=1, lambda =grid)

set.seed(1)
cv.out=cv.glmnet(x,y,alpha=1)
plot(cv.out)
bestlam =cv.out$lambda.min
bestlam #0.009042592

pred.valid.lasso1 <- predict(model.lasso1, newdata = data.valid.std.y)
lasso1.pred=predict(model.lasso1,s=bestlam,newx=newx)
mean((lasso.pred -y.test)^2)

## Principal Component Regression ##

set.seed(1)
pcr.fit=pcr(damt~., data=data.train.std.y, scale=TRUE)
summary (pcr.fit)
# explain 95% of variance with 17 components
# explain 99% with 22

pcr.pred1=predict(pcr.fit,data.valid.std.y,ncomp=17)
mean((pcr.pred1 -y.test)^2)
#1.662192

pcr.pred2=predict(pcr.fit,data.valid.std.y,ncomp=22)
mean((pcr.pred2 -y.test)^2)
#1.564206

pcr.pred3=predict(pcr.fit,data.valid.std.y,ncomp=30)
mean((pcr.pred3 -y.test)^2)
#1.537744
sd((y.valid - pcr.pred3)^2)/sqrt(n.valid.y) 
# 0.1612659

# partial least squares

pls.fit=plsr(damt~., data=data.train.std.y, scale=TRUE)
summary(pls.fit)
# explain 95% of variance with 21 components
# explain 99% of variance with 25 components

pls.pred1=predict(pls.fit,data.valid.std.y,ncomp=21)
mean((pls.pred1 -y.test)^2)
#1.53614
sd((y.valid - pls.pred1)^2)/sqrt(n.valid.y)
# 0.1611788

pls.pred2=predict(pls.fit,data.valid.std.y,ncomp=25)
mean((pls.pred2 -y.test)^2)
#1.537773

pls.pred3=predict(pls.fit,data.valid.std.y,ncomp=30)
mean((pls.pred3 -y.test)^2)
#1.537744

# best model is model model 1 with MSE of 1.536 and SD of 0.161


## Polynomial Regression ##
# Polynomial regression is an extension of linear regression where the predictor variables are 
# raised to a higher power allowing for more flexibility in capturing non-linear relationships (curves)

model.poly1 =lm(damt ~ reg2 + reg3 + reg4 + home + poly(chld,3,raw=T) + poly(hinc,3,raw=T) + poly(genf,3,raw=T) + inca + plow + npro + tgif + lgif +
                  rgif + tdon + t_incm + t_inca + t_agif + t_tgif + t_lgif + t_rgif + t_tdon + t_tlag , 
                data.train.std.y)

model.poly2 =lm(damt ~ reg2 + reg3 + reg4 + home + chld + poly(hinc,2,raw=T) + genf + inca + poly(plow,3,raw=T) + npro + poly(t_tgif,3,raw=T) + poly(t_lgif,3,raw=T) +
                  rgif + tdon + t_incm + t_inca + t_agif + tgif + lgif + t_rgif + t_tdon + t_tlag , 
                data.train.std.y)

model.poly3 =lm(damt ~ reg2 + reg3 + reg4 + home + chld + poly(hinc,2,raw=T) + genf + inca + poly(plow,3,raw=T) + npro + t_tgif + t_lgif +
                  rgif + tdon + t_incm + t_inca + t_agif + tgif + lgif + t_rgif + t_tdon + t_tlag , 
                data.train.std.y)

model.poly4 =lm(damt ~ reg2 + reg3 + reg4 + home + chld + poly(hinc,2,raw=T) + poly(wrat,2,raw=T) + genf + inca + poly(plow,3,raw=T) + npro + t_tgif + t_lgif +
                  rgif + tdon + t_incm + t_inca + t_agif + tgif + lgif + t_rgif + t_tdon + t_tlag , 
                data.train.std.y)

pred.valid.poly3 <- predict(model.poly3, newdata = data.valid.std.y) # validation predictions
mean((y.valid - pred.valid.poly3)^2) # mean prediction error
# 1.486237
sd((y.valid - pred.valid.poly3)^2)/sqrt(n.valid.y) # std error
# 0.1570243

pred.valid.poly4 <- predict(model.poly4, newdata = data.valid.std.y) # validation predictions
mean((y.valid - pred.valid.poly4)^2) # mean prediction error
# 1.344915
sd((y.valid - pred.valid.poly4)^2)/sqrt(n.valid.y) # std error
# 0.1498449

# best model is model 4 with MSE of 1.345 and SD of 0.15


## Step Functions ##
# Step functions split up the range of a variable into regions and fit a piece-wise 
# constant function to each region


model.step1 <- lm(damt ~ reg1 + reg2 + reg3 + reg4 + home + chld + hinc + genf + wrat + 
                  avhv + incm + inca + plow + cut(npro,4) + tgif + lgif + rgif + tdon + tlag + agif, 
                data.train.std.y)
summary(model.step1)
# first two intervals of npro are signficant

model.step1b <- lm(damt ~ reg1 + reg2 + reg3 + reg4 + home + chld + hinc + genf + wrat + 
                  avhv + incm + inca + plow + cut(npro,breaks=c(-0.631,0.704,2.04,3.38)) + tgif + lgif + rgif + tdon + tlag + agif, 
                data.train.std.y)

pred.valid.step1b <- predict(model.step1b, newdata = data.valid.std.y) # validation predictions
mean(pred.valid.step1b, na.rm=TRUE)#14.49534
mean((y.valid - pred.valid.step1b)^2,na.rm=TRUE) # mean prediction error
# 1.92666

model.step2 <- lm(damt ~ reg1 + reg2 + reg3 + reg4 + home + chld + hinc + genf + wrat + 
                  avhv + incm + inca + plow + npro + tgif + lgif + rgif + tdon + tlag + cut(agif,4), 
                data.train.std.y)
summary(model.step2)
# only the first interval of agif is signficant

# apply learnings from step functions and polynomial regression to a generalized additive model
model.gam1 =lm(damt ~ reg2 + reg3 + reg4 + home + chld + poly(hinc,2,raw=T) + poly(wrat,2,raw=T) + genf + inca + poly(plow,3,raw=T) + cut(npro,breaks=c(-0.631,0.704,2.04)) 
		    + t_tgif + t_lgif + rgif + tdon + t_incm + t_inca + t_agif + tgif + lgif + t_rgif + t_tdon + t_tlag , 
                data.train.std.y)

pred.valid.gam1 <- predict(model.gam1, newdata = data.valid.std.y) # validation predictions
mean((y.valid - pred.valid.gam1)^2,na.rm=TRUE) # mean prediction error
# 1.373946
sd((y.valid - pred.valid.gam1)^2)/sqrt(n.valid.y) # std error
# 0.1498449

# GAM model 1 outperforms both step functions, but has slightly higher MSE than the best polynomial regression model


# Splines
# Splines split the range of a variable into regions and fit a polynomial regression model to each region

model.sp1 <- lm(damt ~ reg1 + reg2 + reg3 + reg4 + home + chld + hinc + genf + wrat + 
                  avhv + incm + inca + plow + bs(npro,df=3) + tgif + lgif + rgif + tdon + tlag + agif, 
                data.train.std.y)
summary(model.sp1) # second basis function is significant

pred.valid.sp1 <- predict(model.sp1, newdata = data.valid.std.y) # validation predictions
mean((y.valid - pred.valid.sp1)^2)
# 1.867623

# natural spline

model.ns1 <- lm(damt ~ reg1 + reg2 + reg3 + reg4 + home + chld + hinc + genf + wrat + 
                  avhv + incm + inca + plow + ns(npro,df=3) + tgif + lgif + rgif + tdon + tlag + agif, 
                data.train.std.y)

pred.valid.ns1 <- predict(model.ns1, newdata = data.valid.std.y) # validation predictions
mean((y.valid - pred.valid.ns1)^2)
# 1.866798

# smooth spline

model.ss1 <- smooth.spline(damt ~ reg1 + reg2 + reg3 + reg4 + home + chld + hinc + genf + wrat + npro +
                  avhv + incm + inca + plow + tgif + lgif + rgif + tdon + tlag + agif, 
                data.train.std.y, df=22)


model.gam2 =lm(damt ~ reg2 + reg3 + reg4 + home + chld + poly(hinc,2,raw=T) + poly(wrat,2,raw=T) + genf + inca + poly(plow,3,raw=T) + ns(npro,df=3) 
		    + t_tgif + t_lgif + rgif + tdon + t_incm + t_inca + t_agif + tgif + lgif + t_rgif + t_tdon + t_tlag , 
                data.train.std.y)

pred.valid.gam2 <- predict(model.gam2, newdata = data.valid.std.y) # validation predictions
mean((y.valid - pred.valid.gam2)^2,na.rm=TRUE) # mean prediction error
# 1.344606
sd((y.valid - pred.valid.gam2)^2)/sqrt(n.valid.y) # std error
# 0.1496668

#GAM model 2 applies learnings from natural spline and polynomial regression models
# has MSE and SD on par with best polynomal regression model, but slightly higher


## Local Regression ##

# loess allows a max of 4 predictors, select the best 4 predictor model from model.sub
coef(model.sub,4)
model.lr1 <- loess(damt ~ reg4 + chld + hinc + t_lgif, data.train.std.y, span=.25)
model.lr2 <- loess(damt ~ reg4 + chld + hinc + t_lgif, data.train.std.y, span=.5)
model.lr3 <- loess(damt ~ reg4 + chld + hinc + t_lgif, data.train.std.y, span=.75)


## Decision Trees ##
# Regression trees are like classification trees in that observations are into regions based on a series 
# of decision nodes. A prediction for an observation is the mean of the observations in the region

# simple regression decision tree using all predictors
model.tree1b <- tree(damt ~.,data=data.train.std.y)
summary(model.tree1b)
# "rgif" "lgif" "reg4" "chld" "reg3"
# Residual mean deviance:  1.917 = 3802 / 1984 
# 11 terminal nodes
plot(model.tree1b) + text(model.tree1b)

tree.pred1b=predict(model.tree1b,data.valid.std.y)

mean((y.valid - tree.pred1b)^2,na.rm=TRUE) # mean prediction error
# 2.241075
sd((y.valid - tree.pred1b)^2)/sqrt(n.valid.y) # std error
# 0.1920681

# bagging model

set.seed(1)
model.bag1b= randomForest(damt~.,data=data.train.std.y,mtry=30,importance =TRUE)
model.bag1b
# of squared residuals: 1.483435
# % Var explained: 60.41
importance(model.bag1b)
varImpPlot(model.bag1b)
#indicates most important variables are reg4,rgif,chld,t_rgif,reg3,hinc,t_lgif,lgif,t_agif,wrat

bag.pred1b=predict(model.bag1b,newdata=data.valid.std.y)

mean((y.valid - bag.pred1b)^2,na.rm=TRUE) # mean prediction error
# 1.711384
sd((y.valid - bag.pred1b)^2)/sqrt(n.valid.y) # std error
# 0.1751728

# reduce number of variables tried at each split by 1/3

set.seed(1)
model.bag2b= randomForest(damt~.,data=data.train.std.y,mtry=10,importance =TRUE)
model.bag2b
#of squared residuals: 1.513769
#% Var explained: 59.6
importance(model.bag3)
varImpPlot(model.bag3)

bag.pred2b=predict(model.bag2b,newdata=data.valid.std.y)

mean((y.valid - bag.pred2b)^2,na.rm=TRUE) # mean prediction error
# 1.730375
sd((y.valid - bag.pred2b)^2)/sqrt(n.valid.y) # std error
# 0.1752434

# best model is bag 1 with MSE of 1.711, higher than the polynomial regression model

# boosting models

set.seed(1)
model.boost1b=gbm(damt~.,data=data.train.std.y, distribution="gaussian",n.trees=5000, interaction.depth=5)
summary(model.boost1b)
# most influential: rgif,lgif,agif,reg4,chld,hinc,wrat,reg3,tgif,incm,plow,reg2

boost.pred1b=predict(model.boost1b,newdata=data.valid.std.y,type="response",n.trees=5000)

mean((y.valid - boost.pred1b)^2,na.rm=TRUE) # mean prediction error
# 1.498204
sd((y.valid - boost.pred1b)^2)/sqrt(n.valid.y) # std error
# 0.1659757


model.boost2b=gbm(damt~.,data=data.train.std.y, distribution="gaussian",n.trees=5000, interaction.depth=5,shrinkage =0.2,verbose=F)
summary(model.boost2b)
# most influential: chld, agif, avhv,tgif, hinc, npro, incm, wrat, inca, tdon, reg2, plow, lgif

boost.pred2b=predict(model.boost2b,newdata=data.valid.std.y,type="response",n.trees=5000)

mean((y.valid - boost.pred2b)^2,na.rm=TRUE) # mean prediction error
# 1.969384
sd((y.valid - boost.pred2b)^2)/sqrt(n.valid.y) # std error
# 0.1907212


set.seed(1)
model.boost3b=gbm(damt~.,data=data.train.std.y, distribution="gaussian",n.trees=5000, interaction.depth=10)
summary(model.boost3b)
# most influential: chld, hinc, reg2, home, wrat, tdon, incm, tgif, tlag, reg1

boost.pred3b=predict(model.boost3b,newdata=data.valid.std.y,type="response",n.trees=5000)

mean((y.valid - boost.pred3b)^2,na.rm=TRUE) # mean prediction error
# 1.435618
sd((y.valid - boost.pred3b)^2)/sqrt(n.valid.y) # std error
#  0.1643215

set.seed(1)
model.boost4b=gbm(damt~.,data=data.train.std.y, distribution="gaussian",n.trees=5000, interaction.depth=15)
summary(model.boost4b)
# most influential: chld, hinc, reg2, home, wrat, tdon, incm, tgif, tlag, reg1

boost.pred4b=predict(model.boost3b,newdata=data.valid.std.y,type="response",n.trees=5000)

mean((y.valid - boost.pred4b)^2,na.rm=TRUE) # mean prediction error
# 1.435618
sd((y.valid - boost.pred4b)^2)/sqrt(n.valid.y) # std error
#  0.1643215

# boosting produces the best results from tree methods, but still higher MSE than polynomial regression model


#### RESULTS ####

# Classification Model
# the best performing model from the classification models developed was the third boosting classification tree model

post.test <- predict(model.boost3, data.test.std, type="response",n.trees=5000) # post probabilities for test data

# Oversampling adjustment for calculating number of mailings for test set

n.mail.valid <- which.max(profit.boost3)
tr.rate <- .1 # typical response rate is .1
vr.rate <- .5 # whereas validation response rate is .5
adj.test.1 <- (n.mail.valid/n.valid.c)/(vr.rate/tr.rate) # adjustment for mail yes
adj.test.0 <- ((n.valid.c-n.mail.valid)/n.valid.c)/((1-vr.rate)/(1-tr.rate)) # adjustment for mail no
adj.test <- adj.test.1/(adj.test.1+adj.test.0) # scale into a proportion
n.mail.test <- round(n.test*adj.test, 0) # calculate number of mailings for test set

cutoff.test <- sort(post.test, decreasing=T)[n.mail.test+1] # set cutoff based on n.mail.test
chat.test <- ifelse(post.test>cutoff.test, 1, 0) # mail to everyone above the cutoff
table(chat.test)
#    0    1 
# 1704  303

# Predictive Model
# the best predictive model was the second general additive model with a natural spline

yhat.test <- predict(model.gam2, newdata = data.test.std) # test predictions




# FINAL RESULTS

# Save final results for both classification and regression

length(chat.test) # check length = 2007
length(yhat.test) # check length = 2007
chat.test[1:10] # check this consists of 0s and 1s
yhat.test[1:10] # check this consists of plausible predictions of damt

ip <- data.frame(chat=chat.test, yhat=yhat.test) # data frame with two variables: chat and yhat
write.csv(ip, file="C://Data//charity_results.csv", row.names=FALSE)

