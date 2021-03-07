#Libs
library(tidyverse) # Tibbles
library(readxl) # Import Excel Files
library(Hmisc) # Correlations
library(MASS) # Regression Models
library(pscl) # zeroinfl()
library(performance) # Zero inflation test
library(stargazer) # stargazer()
library(moments)


# Cleaning ----------------------------------------------------------------


setwd("C:\\Users\\Wouter\\OneDrive\\DBI\\Thesis\\R")
full <- as_tibble(read_xlsx("R Dataset.xlsx"))
users <- as_tibble(read_xlsx("User Dataset.xlsx"))

#Remove date errors
full[c(997,998),6] <- as.POSIXct("2019-11-18")
full[1019,6] <- as.POSIXct("2019-10-14")
full[996,6] <- as.POSIXct("2019-10-17")

#Create Tenure
users <- users %>%
  mutate(Tenure = as.numeric(as.POSIXct("2019-11-01") - User_RegistrationDate) / 30.42)

#Create Appreciation
full$Contr_SupportersCnt[is.na(full$Contr_SupportersCnt)] <- 0
full$Contr_LikesCnt[is.na(full$Contr_LikesCnt)] <- 0
grouped <- full %>%
  group_by(User_ID) %>%
  summarise(Appreciation = sum(Contr_SupportersCnt) + sum(Contr_LikesCnt))
users <- cbind(users, grouped[,c(2)])

#Create Pre and Post Period Contributions 
full <- full %>%
  mutate(Period = case_when(
    full$Contr_SubmissionDate <= as.POSIXct("2019-11-01") ~ "P1",
    full$Contr_SubmissionDate > as.POSIXct("2019-11-01") ~ "P2"))

PreCalc <- full %>%
  group_by(User_ID) %>%
  filter(Period == "P1") %>%
  summarise(ActP1_ContrCnt = (length(which(Contr_Type == "Activity"))),
            ConP1_ContrCnt = (length(which(Contr_Type == "Contest"))),
            ProP1_ContrCnt = (length(which(Contr_Type == "Product Idea"))))
            
PostCalc <- full %>%
  group_by(User_ID) %>%
  filter(Period == "P2") %>%
  summarise(ActP2_ContrCnt = (length(which(Contr_Type == "Activity"))),
            ConP2_ContrCnt = (length(which(Contr_Type == "Contest"))),
            ProP2_ContrCnt = (length(which(Contr_Type == "Product Idea"))))

#Merge All Variables
fin <- users[, c(2,30,15,31)]
fin <- merge(fin, PreCalc, by = "User_ID", all.x = TRUE)
fin <- merge(fin, PostCalc, by = "User_ID", all.x = TRUE)
fin[is.na(fin)] <- 0
fin <- fin %>%
  rename(Attention = User_FollowersCnt)
rm(PostCalc,PreCalc,grouped,users)
tasks_act_p1 <- full[full$Contr_Type == "Activity" & full$Period == "P1", c(11,4)]
tasks_act_p2 <- full[full$Contr_Type == "Activity" & full$Period == "P2", c(11,4)]
tasks_con_p1 <- full[full$Contr_Type == "Contest" & full$Period == "P1", c(11,4)]
tasks_con_p2 <- full[full$Contr_Type == "Contest" & full$Period == "P2", c(11,4)]
tasks_act_p1 <- unique(tasks_act_p1)
tasks_act_p2 <- unique(tasks_act_p2)
tasks_con_p1 <- unique(tasks_con_p1)
tasks_con_p2 <- unique(tasks_con_p2)
task <- full[,c(1,2,4,6,11,12,45)]
rm(full)


# Analysis ----------------------------------------------------------------

#Scaling % Centering
fin <- fin %>%
  mutate(Attention = log1p(Attention)) %>%
  mutate(Appreciation = log1p(Appreciation))

#Descriptives
sapply(fin[,2:6], mean)
sapply(fin[,2:6], sd)
summary(fin[,2:6])
sapply(fin[,2:6], skewness)
sapply(fin[,2:6], kurtosis)

rcorr(as.matrix(fin[,-1]), type = "pearson") #Correlations

#Robustness Analysis

fin <- fin %>% 
  mutate(RC_ConP1 = as.factor(case_when(ConP1_ContrCnt > 0 ~ "Y",
                              ConP1_ContrCnt <= 0 ~ "N"))) %>%
  mutate(RC_ProP1 = as.factor(case_when(ProP1_ContrCnt > 0 ~ "Y",
                              ProP1_ContrCnt <= 0 ~ "N"))) %>%
  mutate(RC_ActP2 = as.factor(case_when(ActP2_ContrCnt > 0 ~ "Y",
                              ActP2_ContrCnt <= 0 ~ "N"))) %>%
  mutate(RC_ConP2 = as.factor(case_when(ConP2_ContrCnt > 0 ~ "Y",
                              ConP2_ContrCnt <= 0 ~ "N"))) %>%
  mutate(RC_ProP2 = as.factor(case_when(ProP2_ContrCnt > 0 ~ "Y",
                              ProP2_ContrCnt <= 0 ~ "N")))

#Scaling % Centering
fin <- fin %>%
  mutate(Tenure = scale(Tenure, T, T)) %>%
  mutate(ActP1_ContrCnt = scale(ActP1_ContrCnt, T, T)) %>%
  mutate(ConP1_ContrCnt = scale(ConP1_ContrCnt, T, T)) %>%
  mutate(ProP1_ContrCnt = scale(ProP1_ContrCnt, T, T))
  
  #mutate(Attention = scale(Attention, T, T)) %>%
  #mutate(Appreciation = scale(Appreciation, T, T))

mra_1 <- glm(RC_ConP2 ~ Tenure + Attention + Appreciation + ActP1_ContrCnt + RC_ConP1 + RC_ProP1, data = fin, family = "binomial")
summary(mra_1)

mra_2 <- glm(RC_ProP2 ~ Tenure + Attention + Appreciation + ActP1_ContrCnt + RC_ConP1 + RC_ProP1, data = fin, family = "binomial")
summary(mra_2)

#Hypothesis Testing

m5_2 <- zeroinfl(ConP2_ContrCnt ~ Tenure + Attention + Appreciation + ActP1_ContrCnt + ConP1_ContrCnt + ProP1_ContrCnt | Tenure + Attention + Appreciation + ActP1_ContrCnt + ConP1_ContrCnt + ProP1_ContrCnt,
               data = fin, dist = "negbin")
summary(m5_2)
m5_3 <- zeroinfl(ProP2_ContrCnt ~ Tenure + Attention + Appreciation + ActP1_ContrCnt + ConP1_ContrCnt + ProP1_ContrCnt | Tenure + Attention + Appreciation + ActP1_ContrCnt + ConP1_ContrCnt + ProP1_ContrCnt,
               data = fin, dist = "negbin")
summary(m5_3)

est <- cbind(Estimate = coef(m5_3), confint(m5_3))
exp(est)



# Plots -------------------------------------------------------------------


boxplot(fin$Appreciation)
OutVals = boxplot(fin$Appreciation)$out
length(OutVals)

#Freq Plot
fin %>%
  ggplot(aes(Attention)) +
  geom_histogram(bins = 100) 
#Freq Plot
fin %>%
  ggplot(aes(Appreciation)) +
  geom_histogram(bins = 100) 




sapply(fin[,1:6], skewness)
sapply(fin[,1:6], kurtosis)
sapply(fin[,1:6], sd)
summary(fin[,1:6])

#Freq Plot
full %>%
  ggplot(aes(x = Period, fill = Contr_Type)) +
  geom_bar(position="dodge") +
  labs(x = "Period", y = "", legend = "Contribution Type") +
  theme(text = element_text(size=25)) +
  scale_fill_manual(values=c("#E69F00","#999999","#468189")) +
  guides(fill=guide_legend(title="Contribution Type"))

#Freq Plot
full %>%
  ggplot(aes(x = Contr_Type)) +
  geom_bar(fill = c("#E69F00","#999999","#468189")) +
  labs(x = "Contribution Type", y = "", legend = "Contribution Type") +
  theme(text = element_text(size=25)) +
  scale_fill_manual(values=c("#999999"))

#Freq Plot
fin %>%
  ggplot(aes(ActP1_ContrCnt)) +
  geom_bar() +
  labs(x = "Activity Contributions P2", y = "") +
  theme(text = element_text(size=25))

#Freq Plot
fin %>%
  ggplot(aes(ConP2_ContrCnt)) +
  geom_bar() +
  labs(x = "Contest Contributions P2", y = "") +
  theme(text = element_text(size=25))

#Freq Plot
fin %>%
  ggplot(aes(ProP2_ContrCnt)) +
  geom_bar() +
  labs(x = "Product Idea Contributions P2", y = "") +
  theme(text = element_text(size=25))
theme_light()

#Freq Plot
fin %>%
  ggplot(aes(ActP1_ContrCnt)) +
  geom_bar() +
  labs(x = "x", y = "") +
  theme(text = element_text(size=25))

# Model Comparison --------------------------------------------------------


#Poisson
m1_1 <- glm(ActP2_ContrCnt ~ Tenure + Attention + Appreciation + ActP1_ContrCnt + ConP1_ContrCnt + ProP1_ContrCnt, data = fin, family = poisson) 
m1_2 <- glm(ConP2_ContrCnt ~ Tenure + Attention + Appreciation + ActP1_ContrCnt + ConP1_ContrCnt + ProP1_ContrCnt, data = fin, family = poisson) 
m1_3 <- glm(ProP2_ContrCnt ~ Tenure + Attention + Appreciation + ActP1_ContrCnt + ConP1_ContrCnt + ProP1_ContrCnt, data = fin, family = poisson) 
#Negbin
m2_1 <- glm.nb(ActP2_ContrCnt ~ Tenure + Attention + Appreciation + ActP1_ContrCnt + ConP1_ContrCnt + ProP1_ContrCnt, data = fin) 
m2_2 <- glm.nb(ConP2_ContrCnt ~ Tenure + Attention + Appreciation + ActP1_ContrCnt + ConP1_ContrCnt + ProP1_ContrCnt, data = fin) 
m2_3 <- glm.nb(ProP2_ContrCnt ~ Tenure + Attention + Appreciation + ActP1_ContrCnt + ConP1_ContrCnt + ProP1_ContrCnt, data = fin) 
#Hurdle poisson
m3_1 <- hurdle(ActP2_ContrCnt ~ Tenure + Attention + Appreciation + ActP1_ContrCnt + ConP1_ContrCnt + ProP1_ContrCnt, data = fin, dist = "poisson")
m3_2 <- hurdle(ConP2_ContrCnt ~ Tenure + Attention + Appreciation + ActP1_ContrCnt + ConP1_ContrCnt + ProP1_ContrCnt, data = fin, dist = "poisson")
m3_3 <- hurdle(ProP2_ContrCnt ~ Tenure + Attention + Appreciation + ActP1_ContrCnt + ConP1_ContrCnt + ProP1_ContrCnt, data = fin, dist = "poisson")
#Hurdle negbin
m3_4 <- hurdle(ActP2_ContrCnt ~ Tenure + Attention + Appreciation + ActP1_ContrCnt + ConP1_ContrCnt + ProP1_ContrCnt, data = fin, dist = "negbin")
m3_5 <- hurdle(ConP2_ContrCnt ~ Tenure + Attention + Appreciation + ActP1_ContrCnt + ConP1_ContrCnt + ProP1_ContrCnt, data = fin, dist = "negbin")
m3_6 <- hurdle(ProP2_ContrCnt ~ Tenure + Attention + Appreciation + ActP1_ContrCnt + ConP1_ContrCnt + ProP1_ContrCnt, data = fin, dist = "negbin")
# Zerinfl negbin
m5_1 <- zeroinfl(ActP2_ContrCnt ~ Tenure + Attention + Appreciation + ActP1_ContrCnt + ConP1_ContrCnt + ProP1_ContrCnt | Tenure + Attention + Appreciation + ActP1_ContrCnt + ConP1_ContrCnt + ProP1_ContrCnt, data = fin, dist = "negbin")
m5_2 <- zeroinfl(ConP2_ContrCnt ~ Tenure + Attention + Appreciation + ActP1_ContrCnt + ConP1_ContrCnt + ProP1_ContrCnt | Tenure + Attention + Appreciation + ActP1_ContrCnt + ConP1_ContrCnt + ProP1_ContrCnt, data = fin, dist = "negbin")
m5_3 <- zeroinfl(ProP2_ContrCnt ~ Tenure + Attention + Appreciation + ActP1_ContrCnt + ConP1_ContrCnt + ProP1_ContrCnt | Tenure + Attention + Appreciation + ActP1_ContrCnt + ConP1_ContrCnt + ProP1_ContrCnt, data = fin, dist = "negbin")
# Zeroinfl poisson
m5_4 <- zeroinfl(ActP2_ContrCnt ~ Tenure + Attention + Appreciation + ActP1_ContrCnt + ConP1_ContrCnt + ProP1_ContrCnt | Tenure + Attention + Appreciation + ActP1_ContrCnt + ConP1_ContrCnt + ProP1_ContrCnt, data = fin, dist = "poisson")
m5_5 <- zeroinfl(ConP2_ContrCnt ~ Tenure + Attention + Appreciation + ActP1_ContrCnt + ConP1_ContrCnt + ProP1_ContrCnt | Tenure + Attention + Appreciation + ActP1_ContrCnt + ConP1_ContrCnt + ProP1_ContrCnt, data = fin, dist = "poisson")
m5_6 <- zeroinfl(ProP2_ContrCnt ~ Tenure + Attention + Appreciation + ActP1_ContrCnt + ConP1_ContrCnt + ProP1_ContrCnt | Tenure + Attention + Appreciation + ActP1_ContrCnt + ConP1_ContrCnt + ProP1_ContrCnt, data = fin, dist = "poisson")

#Dispersion Test
check_overdispersion(m1_1)
check_overdispersion(m1_2)
check_overdispersion(m1_3)

#Zeros test
check_zeroinflation(m1_1)
check_zeroinflation(m1_2)
check_zeroinflation(m1_3)

stargazer(m1_1, m1_2, m1_3, m2_1, m2_2, m2_3,  type = 'text', star.cutoffs = c(.05, .01, .001), 
          no.space = T, digits = 3)
stargazer(m3_1, m3_2, m3_3, m3_4, m3_5, m3_6,  type = 'text', star.cutoffs = c(.05, .01, .001), 
          no.space = T, digits = 3)
stargazer(m5_1, m5_2, m5_3, m5_4, m5_5, m5_6,  type = 'text', star.cutoffs = c(.05, .01, .001), 
          no.space = T, digits = 3)


