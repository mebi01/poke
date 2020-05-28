# What Makes a Pokémon Legendary?
# data camp
# may 27, 2020

# Not all Pokémon are created equal. Some are consigned to mediocrity, useless in battle until they reach their more evolved states. Others – like Zapdos, Articuno and Moltres – are so unique and powerful that they have officially been classified as legendary.
# 
# But what exactly makes a Pokémon the stuff of legend? In this project, we will answer that question with the help of a dataset that includes the base stats, height, weight and type of 801 Pokémon from all seven generations. Using the random forest algorithm, we will predict Pokemon status based on these characteristics and rank their importance in determining whether a Pokemon is classified as legendary.
# 
# Students should be familiar with the tidyverse suite of packages, particularly ggplot2 for data visualization and dplyr for data manipulation. They should also have experience with classification problems and tree-based methods, as taught through Supervised Learning in R: Classification and Machine Learning with Tree-Based Models in R.
# 
# This project uses a subset of The Complete Pokemon Dataset published on Kaggle.


# In the world of Pokémon academia, one name towers above any other – Professor Samuel Oak. While his colleague Professor Elm specializes in Pokémon evolution, Oak has dedicated his career to understanding the relationship between Pokémon and their human trainers. A former trainer himself, the professor has first-hand experience of how obstinate Pokémon can be – particularly when they hold legendary status.
# 
# For his latest research project, Professor Oak has decided to investigate the defining characteristics of legendary Pokémon to improve our understanding of their temperament. Hearing of our expertise in classification problems, he has enlisted us as the lead researchers.
# 
# Our journey begins at the professor's research lab in Pallet Town, Kanto. The first step is to open up the Pokédex, an encyclopaedic guide to 801 Pokémon from all seven generations.


# load packages
lapply(c("ggplot2", "dplyr", "tidyverse"), require, character.only=T)

poke <- read.csv("https://raw.githubusercontent.com/kelandrin/What-makes-pokemon-legendary/master/datasets/pokedex.csv")
write.csv(poke, "poke.csv")
str(poke)
# need to convert some variables to factors
poke$is_legendary <- as.factor(poke$is_legendary)



############################ 2 How many Pokémon are legendary?
#   After browsing the Pokédex, we can see several variables that could feasibly explain what makes a Pokémon legendary. We have a series of numerical fighter stats – attack, defense, speed and so on – as well as a categorization of Pokemon type (bug, dark, dragon, etc.). is_legendary is the binary classification variable we will eventually be predicting, tagged 1 if a Pokémon is legendary and 0 if it is not.
# Before we explore these variables in any depth, 
# let's find out how many Pokémon are legendary out of the 801 total, using the handy count() function from the dplyr package.

is_leg <- poke %>% count(is_legendary)%>% mutate(prop= n/sum(n))


############################  3 Legendary Pokémon by height and weight
# We now know that there are 70 legendary Pokémon – a sizable minority at 9% of the population! Let's start to explore some of their distinguishing characteristics.
# 
# First of all, we'll plot the relationship between height_m and weight_kg for all 801 Pokémon,
#highlighting those that are classified as legendary. We'll also add conditional labels to the plot, 
# which will only print a Pokémon's name if it is taller than 7.5m or heavier than 600kg.


poke %>%
  ggplot(aes(height_m, weight_kg))+
  geom_point(aes(col= is_legendary))+
  geom_text( aes(label= ifelse(test= height_m > 7.5 |weight_kg >600, 
                               yes= as.character(name),
                               no= "")),
             vjust=0, hjust=0)+
  expand_limits(x=20, y= 1200)+ 
  geom_smooth(method="lm", se= F, linetype="dashed") + # SE= T removes surronding margines
  labs(title= 'Weight and Height',
       x= " Height",
       y="Weight")

############################  4. Legendary Pokémon by type
# It seems that legendary Pokémon are generally heavier and taller, but with many exceptions. For example, Onix (Gen 1), Steelix (Gen 2) and Wailord (Gen 3) are all extremely tall, but none of them have legendary status. 
# There must be other factors at play.
# We will now look at the effect of a Pokémon's type on its legendary/non-legendary classification. 
# There are 18 possible types, ranging from the common (Grass / Normal / Water)
# to the rare (Fairy / Flying / Ice). We will calculate the proportion of legendary Pokémon
#within each category, and then plot these proportions using a simple bar chart.

n_leg_type<- poke %>% group_by(type) %>% mutate(n_leg= as.numeric(is_legendary)-1) %>% #changing factor to numeric changes 0 to 1 and 1 to 2
  
  summarise(mean_leg= mean(n_leg)) %>% ungroup()


n_leg_type %>% ggplot(aes(reorder(type, -mean_leg), mean_leg))+ geom_bar(stat="identity")+ coord_flip()

############################   5. Legendary Pokémon by fighter stats
# # There are clear differences between Pokémon types in their relation to legendary status. 
# While more than 30% of flying and psychic Pokémon are legendary, there is no such thing as 
# a legendary poison or fighting Pokémon!
# #   
# # Before fitting the model, we will consider the influence of a Pokémon's fighter stats (attack, defense, etc.) 
#   on its status. Rather than considering each stat in isolation, we will produce a boxplot for all of them 
# simultaneously using the facet_wrap() function.

# get box plot of is-legen data separated for each variable (attach, defense,...). To get this plot, 

# we need long data one col is_legn, one col for key of the other variables, and the last col, value of each var 

poke_long<- poke%>% select(is_legendary, attack, sp_attack, defense, sp_defense, hp, speed)%>%
  gather(key= fighter_stat, value= value, -is_legendary)

poke_long %>% 
  ggplot(aes(is_legendary, value, fill= is_legendary))+
  geom_boxplot()+ 
  facet_wrap(~fighter_stat)+
  theme_minimal()


################################## 7. Fit a decision tree
# Now we have our training and test sets, we can go about building our classifier. But before we fit a random forest, we will fit a simple classification decision tree. This will give us a baseline fit against which to compare the results of the random forest, as well as an informative graphical representation of the model.
# 
# Here, and also in the random forest, we will omit incomplete observations by setting the na.action argument to na.omit. This will remove a few Pokémon with missing values for height_m and weight_kg from the training set. Remember the warning messages when we made our height/weight plot in Task 3? These are the Pokémon to blame!
set.seed(1234)
n_row= nrow(poke)
sample_n<- sample (n_row, .6*n_row)

train_poke <- poke %>% filter (row_number() %in% sample_n)
test_poke <- poke %>% filter(!row_number() %in% sample_n)


install.packages("rpart", "rpart.plot")
lapply(c("rpart", "rpart.plot"), require, character.only=T)
set.seed(1234)
poke_tree= rpart(is_legendary~ attack + defense + height_m + 
                   hp + sp_attack + sp_defense + speed + type + weight_kg,
                 data= train_poke,
                 method="class",
                 na.action = na.omit)
rpart.plot(poke_tree)


################################# 8. Fit a random forest
# Each node of the tree shows the predicted class, the probability of being legendary, and the percentage of Pokémon in that node. The bottom-left node, for example – for those with sp_attack < 118 and weight_kg < 169 – represents 84% of Pokémon in the training set, predicting that each only has a 3% chance of being legendary.
# 
# Decision trees place the most important variables at the top and exclude any they don't find to be useful. In this case, sp_attack occupies node 1 while attack, defense, sp_defense and height_m are all excluded.
# 
# However, decision trees are unstable and sensitive to small variations in the data. It therefore makes sense to fit a random forest – an ensemble method that averages over several decision trees all at once. This should give us a more robust model that classifies Pokémon with greater accuracy.
install.packages("randomForest")
library(randomForest)

poke_randomF<- randomForest(is_legendary~attack + defense + height_m + 
                              hp + sp_attack + sp_defense + speed + type + weight_kg,
                            data= train_poke,
                            importance= TRUE,
                            na.action = na.omit)
print(poke_randomF)


########################## 9. Assess model fit
# Looking at the model output, we can see that the random forest has an out-of-bag (OOB) error of 7.48%, which isn't bad by most accounts. However, since there are 24 true positives and 24 false negatives, the model only has a recall of 50%, which means that it struggles to successfully retrieve every legendary Pokémon in the dataset.
# In order to allow direct comparison with the decision tree, we will plot the ROC curves for both models using the ROCR package, which will visualize their true positive rate (TPR) and false positive rate (FPR) respectively. The closer the curve is to the top left of the plot, the higher the area under the curve (AUC) and the better the model.
install.packages("ROCR")
library(ROCR)
prob_tree<- predict(poke_tree, test_poke, type="prob")
pred_tree <- prediction(prob_tree[, 2], test_poke$is_legendary)
perf_tree <- performance(pred_tree, "tpr", "fpr")

# RF
prob_rf <- predict(poke_randomF, test_poke, type="prob")
pred_rf<- prediction(prob_rf[,2], test_poke$is_legendary)
perf_rf <- performance(pred_rf, "tpr", 'fpr')
plot(perf_tree,  col= "red")
plot(perf_rf, col= "blue")


#######################  10. Analyze variable importance
# It's clear from the ROC curves that the random forest is a substantially better model, boasting an AUC (not calculated above) of 91% versus the decision tree's 78%. When calculating variable importance, it makes sense to do so with the best model available, so we'll use the random forest for the final part of our analysis.
# 
# Note that a random forest returns two measures of variable importance:
# 
# MeanDecreaseAccuracy – how much the model accuracy suffers if you leave out a particular variable
# MeanDecreaseGini – the degree to which a variable improves the probability of an observation being classified one way or another (i.e. 'node purity').
# Together, these two measures will allow us to answer our original research question – what makes a Pokémon legendary?

importance_forest <- importance(poke_randomF)
importance_forest
# Create a dotchart of variable importance
varImpPlot_forest <-varImpPlot(poke_randomF)

################################### 11. Conclusion
# According to the variable importance plot, sp_attack is the most important factor in determining whether or not a Pokémon is legendary, followed by speed. The plot doesn't tell us whether the variables have a positive or a negative effect, but we know from our exploratory analysis that the relationship is generally positive. We therefore conclude that legendary Pokémon are characterized primarily by the power of their special attacks and secondarily by their speediness, while also exhibiting higher fighting abilities across the board.
# 
# Congratulations on completing your research into legendary Pokémon – Professor Oak is excited to share the findings! To finish, we'll answer a few of his questions about the variable importance results.
