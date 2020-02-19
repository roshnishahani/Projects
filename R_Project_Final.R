# Load Packages
library(ggplot2)
library(tidyverse)
library(psych)
library(MASS)
library(MLmetrics)

# Load data
bikeDay <- read.csv("day.csv")
bikeHour <- read.csv("hour.csv")

# look at data
head(bikeDay)
head(bikeHour)

# details about data
describe(bikeDay)

# check missing values
sum(is.na(bikeDay))
sum(is.na(bikeHour))

# Removing some variables that are not used in the model
bikeDay <- bikeDay %>% dplyr::select(-c(instant,dteday,casual,registered))

# Examine influence of year
ggplot(bikeDay, aes(x = cnt)) + geom_histogram(color="darkblue", fill="lightblue") + xlab("Demand") + ggtitle("Histogram of Demand") + facet_wrap(~yr)
# There is a significant increase in demand from 2011 to 2012
# However, in 2011 the company was still newly launched 
# and we do not expect this growth in customer base to persist in the future
# Therefore, to have a better understanding of the influence from environmental 
# and seasonal settings on demand, and to better predict future demand
# We exclude data from the first year
bikeDay <- subset(bikeDay, yr != 0)
bikeDay <- bikeDay %>% dplyr::select(-c(yr))

# Get the names
names(bikeDay)

# Assign factors to categorical variables
bikeDay$season <- factor(bikeDay$season)
bikeDay$mnth <- factor(bikeDay$mnth)
bikeDay$holiday <- factor(bikeDay$holiday)
bikeDay$weekday <- factor(bikeDay$weekday)
bikeDay$workingday <- factor(bikeDay$workingday)
bikeDay$weathersit <- factor(bikeDay$weathersit)

# Check the structure
str(bikeDay)

# Correlation Matrix
cor(bikeDay[,c(7,8,9,10,11)])
pairs(bikeDay[,c(7,8,9,10,11)])

# Boxplots for categorical variables
ggplot(bikeDay, aes(x = season, y = cnt, fill = season)) + geom_boxplot() + ggtitle("Boxplot by Season") + theme(legend.position="none") 
ggplot(bikeDay, aes(x = mnth, y = cnt, fill = mnth)) + geom_boxplot() + ggtitle("Boxplot by Month") + theme(legend.position="none")
ggplot(bikeDay, aes(x = holiday, y = cnt, fill = holiday)) + geom_boxplot() + ggtitle("Boxplot by Holiday") + theme(legend.position="none")
ggplot(bikeDay, aes(x = weekday, y = cnt, fill = weekday)) + geom_boxplot() + ggtitle("Boxplot by Weekday") + theme(legend.position="none")
ggplot(bikeDay, aes(x = workingday, y = cnt, fill = workingday)) + geom_boxplot() + ggtitle("Boxplot by Working Day") + theme(legend.position="none")
ggplot(bikeDay, aes(x = weathersit, y = cnt, fill = weathersit)) + geom_boxplot() + ggtitle("Boxplot by Weather Situation") + theme(legend.position="none")
# According to boxplots， count seems most influenced by Season, Month and Weather Situation

# Histograms for numeric variables
ggplot(bikeDay, aes(x = cnt)) + geom_histogram(color="darkblue", fill="lightblue") + xlab("Demand") + ggtitle("Histogram of Demand")
ggplot(bikeDay, aes(x = temp)) + geom_histogram(color="orange", fill="yellow") + xlab("Temperature") + ggtitle("Histogram of Temperature")
ggplot(bikeDay, aes(x = hum)) + geom_histogram(color="darkblue", fill="lightblue") + xlab("Humidity") + ggtitle("Histogram of Humidity")
ggplot(bikeDay, aes(x = windspeed)) + geom_histogram(color="orange", fill="yellow") + xlab("Wind Speed") + ggtitle("Histogram of Wind Speed")

# Remove atemp which highly correlates with temp
bikeDay <- bikeDay %>% dplyr::select(-atemp)

# There is one day with 0 humidity which is impossible so we remove the row
bikeDay <- subset(bikeDay, hum != 0)

# Explore the interactions
# season vs temp
pt_ssn_tmp = ggplot(data=bikeDay, aes(x=temp, y=cnt, color=season)) + 
  geom_point() + 
  scale_color_brewer(palette="Set1") +
  facet_wrap(~season)
pt_ssn_tmp

ssntmp.lm <- lm(cnt ~ as.factor(season) + temp + 
                  season:temp, data = bikeDay)

qplot(x = temp, y = cnt, color = season, data = bikeDay) +
  stat_smooth(method = "lm", se = FALSE, fullrange = TRUE) + 
  scale_color_brewer(palette="Set1")

# weather vs temp
pt_wtr_tmp = ggplot(data=bikeDay, aes(x=temp, y=cnt, color=weathersit)) + 
  geom_point() + facet_wrap(~weathersit) + 
  scale_color_brewer(palette="Set1")
pt_wtr_tmp

wtrtmp <- lm(cnt ~ season + temp + 
               season:temp, data = bikeDay)

qplot(x = temp, y = cnt, color = weathersit, data = bikeDay) +
  stat_smooth(method = "lm", se = FALSE, fullrange = TRUE) + 
  scale_color_brewer(palette="Set1")


# holiday vs temp
pt_hld_tmp = ggplot(data=bikeDay, aes(x=temp, y=cnt, color=holiday)) + 
  geom_point() + facet_wrap(~holiday) + 
  scale_color_brewer(palette="Set1")
pt_hld_tmp

hldtmp.lm <- lm(cnt ~ holiday + temp + 
                  holiday:temp, data = bikeDay)

qplot(x = temp, y = cnt, color = holiday, data = bikeDay) +
  stat_smooth(method = "lm", se = FALSE, fullrange = TRUE) + 
  scale_color_brewer(palette="Set1")

# Implement Linear Model
blm<- lm(cnt~.,data = bikeDay)
summary(blm)

# Implement Linear Model with Log Transformation
logblm <- lm(log(cnt)~.,data = bikeDay)
summary(logblm)

# Intuitive Model Selection
intlm <- lm(cnt~season + holiday + weathersit + temp, data = bikeDay)
summary(intlm)
AIC(intlm)

# Backward Elimination
step <- stepAIC(blm, direction="backward")
step$anova

# Best Model with Backward Elimination
bikelm <- lm(cnt ~ season + mnth + holiday + weekday + weathersit + temp + hum + 
               windspeed, data = bikeDay)
summary(bikelm)

# Testing Linear Model
newdata <- bikeDay %>% select_("season","mnth","holiday","weekday","weathersit","temp","hum","windspeed")
predictions <- predict(bikelm,newdata = newdata)
head(predictions)

# Plot Observed vs Predicted
plot(bikeDay$cnt,type='l',col="blue",main = "Observed Vs Predicted",ylab = "Daily Demand")
lines(predictions,col="red")
par(xpd=TRUE)
legend(0, -2500, legend=c("Observed", "Predicted"),col=c("blue", "red"),lty = 1,box.lty = 0)

# Histogram of Residuals
hist(predictions-bikeDay$cnt,col="steelblue",main = "Histogram of residuals",xlab = "Predicted-Observed")

# Scatterplot of Demand Vs Prediction
plot(bikeDay$cnt,predictions,xlab="Demand",ylab="Prediction",main="Scatterplot of Demand Vs Prediction",col="tomato",pch=15)

# Check root-mean-square error
RMSE(y_pred =predictions ,y_true = bikeDay$cnt)
