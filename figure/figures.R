library("reshape2")
library("ggplot2")
setwd("G:\\NYU_Class\\DL\\project\\figure")
df <- read.csv(file="train_acc.csv", header=TRUE, sep=",")
ggplot(df, aes(Epoch)) + 
  labs(y="Training accuracy",x="Epoch") + 
  geom_line(aes(y = Sort.CNN.LSTM, colour = "Sort-CNN-LSTM"),size=1.5) + 
  geom_line(aes(y = MLP, colour = "MLP"),size=1.5) + 
  geom_line(aes(y = CNN, colour = "CNN"),size=1.5) + 
  geom_line(aes(y = LSTM, colour = "LSTM"),size=1.5) +
  ylim(0.2,1)
df2 <- read.csv(file="sort_acc.csv", header=TRUE, sep=",")
ggplot(df2, aes(Epoch)) + 
  labs(y="Training accuracy",x="Epoch") + 
  geom_line(aes(y = sort, colour = "Sort-CNN-LSTM"),size=1.5) + 
  geom_line(aes(y = unsort, colour = "CNN-LSTM (no sort)"),size=1.5) + 
  ylim(0.2,1)
df3 <- read.csv(file="batch_Size.csv", header=TRUE, sep=",")
ggplot(df3, aes(x = Sample_Size, y = Accuracy )) + 
  geom_line(size=1.5) + geom_point(size=4) + 
  labs(y="Validation accuracy",x="Sample size") +
  ylim(0.15,0.7) + theme_bw() +
  
  #eliminates background, gridlines, and chart border
  theme(
    text = element_text(size=20),
    plot.background = element_blank()
    ,panel.grid.major = element_blank()
    ,panel.grid.minor = element_blank()
    ,panel.border = element_blank()
  ) +
  
  #draws x and y axis line
  theme(axis.line = element_line(color = 'black'))