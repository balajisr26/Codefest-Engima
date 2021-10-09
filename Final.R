library(data.table)

#Read Input Files Namely User, Problem and Train Dataset
user<-fread('user_data.csv',data.table=F,stringsAsFactors = F)
train<-fread('train_submissions.csv',data.table=F,stringsAsFactors = F)
problem<-fread('problem_data.csv',data.table=F,stringsAsFactors = F)
train<-merge(train,problem,all.x=T,by="problem_id")
train<-merge(train,user,by="user_id")

#Read Test Files
test<-fread('test_submissions_NeDLEvX.csv',data.table=F,stringsAsFactors = F)
test<-merge(test,problem,all.x=T,by="problem_id")
test<-merge(test,user,by="user_id")

#Bag of Words for Tags Column (#Feature Engineering 1)
library(tm)
library(SnowballC)

Corpus<-  Corpus(VectorSource(c(train$tags, test$tags)))
Corpus<- tm_map(Corpus, tolower)
#Corpus<-tm_map(Corpus, PlainTextDocument)
#Corpus<- tm_map(Corpus, removePunctuation)
#Corpus<- tm_map(Corpus, removeWords, stopwords("english"))
#Corpus<- tm_map(Corpus, stemDocument,language="english")

#corpus<-Corpus(VectorSource(Corpus))
dtm<- DocumentTermMatrix(Corpus)

#sparse<- removeSparseTerms(dtm, 0.995)

tags = as.data.frame(as.matrix(dtm))

colnames(tags) = make.names(colnames(tags))

str(tags)

trainds<-head(tags, nrow(train))
testds<-tail(tags,nrow(test))

str(trainds)

str(train)

trainds$user_id<-as.factor(train$user_id)
trainds$problem_id<-as.factor(train$problem_id)
trainds$attempts_range<-train$attempts_range
trainds$level_type<-as.factor(train$level_type)
trainds$points<-train$points
trainds$submission_count<-train$submission_count
trainds$problem_solved<-train$problem_solved
trainds$contribution<-train$contribution
trainds$country<-as.factor(train$country)
trainds$follower_count <-train$follower_count
trainds$last_online_time_seconds<-train$last_online_time_seconds
trainds$max_rating<-train$max_rating
trainds$rating<-train$rating
trainds$rank<-as.factor(train$rank)
trainds$registration_time_seconds<-train$registration_time_seconds
trainds$tags<-as.factor(train$tags)

testds$user_id<-as.factor(test$user_id)
testds$problem_id<-as.factor(test$problem_id)
testds$attempts_range<-test$attempts_range
testds$level_type<-as.factor(test$level_type)
testds$points<-test$points
testds$submission_count<-test$submission_count
testds$problem_solved<-test$problem_solved
testds$contribution<-test$contribution
testds$country<-as.factor(test$country)
testds$follower_count <-test$follower_count
testds$last_online_time_seconds<-test$last_online_time_seconds
testds$max_rating<-test$max_rating
testds$rank<-as.factor(test$rank)
testds$rating<-test$rating
testds$registration_time_seconds<-test$registration_time_seconds
testds$tags<-as.factor(test$tags)

#Feature Engineering 2
trainds$sub_sol<-trainds$submission_count/trainds$problem_solved
trainds$rankdiff<-trainds$max_rating-trainds$rating

#Feature Engineering 3
testds$sub_sol<-testds$submission_count/testds$problem_solved
testds$rankdiff<-testds$max_rating-testds$rating

#Feature Engineering 4
trainds$secondsdiff<-trainds$last_online_time_seconds-trainds$registration_time_seconds
testds$secondsdiff<-testds$last_online_time_seconds-testds$registration_time_seconds

#Feature Engineering 5
trainds$level_rank<-as.factor(paste0(trainds$level_type,trainds$rank))
testds$level_rank<-as.factor(paste0(testds$level_type,testds$rank))

# Remove Column ID for model training purposesS
features<-setdiff(names(trainds),c('ID','attempts_range'))

#Library Catboost
library(catboost)

# Convert training data into catboost internal format from dataframe
pool<-catboost.from_data_frame(data=trainds[,features],target = trainds$attempts_range)

testds$attempts_range<--99


# Convert testing data into catboost internal format from dataframe
testpool<-catboost.from_data_frame(data=testds[,features],target = testds$attempts_range)

#Params for Catboost
fit_params <- list(iterations = 1000,
                   thread_count = 2,
                   loss_function = 'MultiClass',
                   eval_metric='MultiClass',
                   depth = 5,
                   learning_rate = 0.3,
                   #   l2_leaf_reg = 3.5,
                   overfitting_detector_type='IncToDec',
               #    od_wait=30,
                   random_seed=88,
                   verbose=TRUE
                    #use_best_model=TRUE
                   #train_dir = 'train_dir'
)
#Train the Model
model <- catboost.train(pool,test_pool = NULL,params = fit_params)
#Get Variable Importance
catboost.importance(model,pool)
#Get Prediction
prediction <- catboost.predict(model, testpool,type="Class")
df<-data.frame(ID=test$ID,attempts_range=prediction)
#Create File for Submission
write.csv(df,'3_7.csv',row.names = F)
