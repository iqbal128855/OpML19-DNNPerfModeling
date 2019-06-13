# load library
library(MASS)

# read data
file<-"./TX2/data/tx2_sampled_output_inceptionv3_800x800.csv"
df<-read.csv(file)

# perform stepwise regression
stepwise_regression <- function(data,s) {
    if (s=="inference_time") {
        Y<-data$inference_time
    } else if (s=="power_consumption") {
        Y<-data$power_consumption
    } else
        return
    X<-subset(data,select=c(7:15))
    model<-lm(Y ~ .,data=X)
    stepwise_model=step(model, scope= . ~ . ^9, direction= 'both')
    return (stepwise_model$coefficients)
    
}

# inference time 
itm=stepwise_regression(df,"inference_time")
# power consumption
pcm=stepwise_regression(df,"power_consumption")

# display
cat("+++++Inference Time+++++\n")
print(itm)
cat("+++++Power Consumption+++++\n")
print(pcm)
