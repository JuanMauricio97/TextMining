
# Instalación de librerías y paquetes necesarios
install.packages(c("ggplot2", "e1071", "caret", "quanteda", "irlba", "randomForest", "httr", "ROAuth", "tm", "SnowballC",
				   "caTools", "plyr", "wordcloud", "tidytext", "tidyverse", "topicmodels", "corpus", "kernlab", "doSNOW",
				   "hashmap", "text2vec", "Matrix", "ROCR", "pROC"))

library(ggplot2)
library(e1071)
library(caret)
library(quanteda)
library(irlba)
library(randomForest)
library(httr)
library(ROAuth)
library(tm)
library(SnowballC)
library(caTools)
library(plyr)
library(wordcloud)
library(tidytext)
library(tidyverse)
library(topicmodels)
library(corpus)
library(doSNOW)
library(kernlab)
library(hashmap)
library(text2vec)
library(Matrix)
library(udpipe)
library(ROCR)
library(pROC)

# --------------------------------------------------------------------------------------------------------------------------------


# Fijar directorio de trabajo (escoger la ubicación que contenga la carpeta /corpus donde está el archivo documents.txt para trabajar)
setwd(".. /")


# Se crean los nombres de las columnas para visualización del corpus en un data frame
columnas <- c("label", "description")

# Se lee el archivo que contiene los datos; el parámetro stringsAsFactors indica que cuando lea cadenas de texto
# no los convierta en factores, R lo hace por defecto
data <- read.table(".. /corpus/documents.txt", header = TRUE, sep = "\t", stringsAsFactors = FALSE, col.names = columnas)
View(data)


# Verificamos si hay datos perdidos; lo óptimo para continuar es que sea 0, ya que indica que los datos están
# completos
length(which(!complete.cases(data)))

# Convertimos nuestra clase label en un factor
data$label <- as.factor(data$label)

# Entonces, lo primero que se hace es explorar los datos
# Primero, revisamos la distribución de las etiquetas clase [0 (error de usuario) vs. 1 (bug del software)]
# table(data$label) nos indica cuántos datos hay de cada clase y prop.table nos muestra la proporción
prop.table(table(data$label))

# Ahora, veamos la distribución de las longitudes del texto de las descripciones
# nchar cuenta el número de caracteres y lo guarda en la nueva característica TextLength
# summary produce un resumen de resultados del modelo
data$TextLength <- nchar(data$description)
summary(data$TextLength)

# Visualizamos la disribución, añadiendo la segmentación de la forma 0/1
ggplot(data, aes(x = TextLength, fill = label)) + theme_bw() + geom_histogram(binwidth = 10) +
			labs(y = "Conteo del texto", x = "Longitud del texto",
				title = "Distribución de las longitudes del texto con la clase label")

# --------------------------------------------------------------------------------------------------------------------------------

#                                                          ***** CLEANNING DATA *****

# Procedemos a limpiar la información realizando lemmatization, generando tokenization, y capitalización, considerando
# puntuaciones, símbolos y stop words y, finalmente, generamos una DFM para ello, utilizamos la librería Quanteda

# Tokenization y eliminación del "ruido"
# Primeramente, le decimos a R, con el parámetro what de la función tokens de Quanteda, que de cada palabra haga un token
# y posteriormente, que remueva números, signos de puntuación, símbolos


# Función para lemmatization
cols <- c("lemma", "word")

# Ya en el directorio de trabajo fijado anteriormente, escoger la carpeta /lemmas donde está el archivo lemmatization.txt
# para poder aplicar la lemmatization a los términos de cada documento
lemas <- read.table(".. /lemmas/lemmatization.txt", header = TRUE, sep = "\t", stringsAsFactors = FALSE, col.names = cols)
View(lemas)
lemma_hm = hashmap(lemas$word, lemas$lemma)

lemma_tokenizer = function(x, lemma_hashmap, 
                           tokenizer = text2vec::word_tokenizer) {
  tokens_list = tokenizer(x)
  for(i in seq_along(tokens_list)) {
    tokens = tokens_list[[i]]
    replacements = lemma_hashmap[[tokens]]
    ind = !is.na(replacements)
    tokens_list[[i]][ind] = replacements[ind]
  }
  tokens_list
}

# Capitalización y lemmatization
dataTokens <- tokens_tolower(as.tokens(lemma_tokenizer(data$description, lemma_hm)))
# dataTokens <- tokens_tolower(dataTokens)
dataTokens[23]

dataTokens <- tokens(dataTokens, what = "word", remove_numbers = TRUE, remove_punct = TRUE, remove_symbols = TRUE,
            remove_hyphens = TRUE)

dataTokens[23]

# Se eliminan los stop words
dataTokens <- tokens_select(dataTokens, stopwords("spanish"), selection = "remove")
dataTokens[23]


# Se crea la matriz de frecuencia de los documentos, pero como R la transforma primeramente en un objeto, se utiliza la función
# as.matrix() para hacerla más manipulable visualmente
dataTokensDFM <- dfm(dataTokens)
dataTokensMatrix <- as.matrix(dataTokensDFM)
View(dataTokensMatrix)

# --------------------------------------------------------------------------------------------------------------------------------

#                                                            ***** WORDCLOUDS *****

# Si quisiéramos visualizar de mejor manera la frecuencia de las palabras en los conjuntos de datos del training y tests,
# podríamos verlos con nubes de palabras o Wordclouds

# Dividimos la información

set.seed(3298)

# Asignamos la proporción para los conjuntos de datos
info <- createDataPartition(data$label, times = 1, p = 0.8, list = FALSE)

training <- data[indices,]
tests <- data[-indices,]

trainingTokens <- tokens_tolower(as.tokens(lemma_tokenizer(training$description, lemma_hm)))
trainingTokens[23]

trainingTokens <- tokens(trainingTokens, what = "word", remove_numbers = TRUE, remove_punct = TRUE, remove_symbols = TRUE,
            remove_hyphens = TRUE)

trainingTokens[23]

trainingTokens <- tokens_select(trainingTokens, stopwords("spanish"), selection = "remove")
trainingTokens[23]

trainingTokensDFM <- dfm(trainingTokens)
trainingTokensMatrix <- as.matrix(trainingTokensDFM)
View(trainingTokensMatrix)



testsTokens <- tokens_tolower(as.tokens(lemma_tokenizer(tests$description, lemma_hm)))
testsTokens[23]

testsTokens <- tokens(testsTokens, what = "word", remove_numbers = TRUE, remove_punct = TRUE, remove_symbols = TRUE,
            remove_hyphens = TRUE)
testsTokens[23]

testsTokens <- tokens_select(testsTokens, stopwords("spanish"), selection = "remove")
testsTokens[23]

testsTokensDFM <- dfm(testsTokens)
testsTokensMatrix <- as.matrix(testsTokensDFM)
View(testsTokensMatrix)


# Wordclouds para el corpus, el training y el tests

freq = data.frame(sort(colSums(as.matrix(dataTokensMatrix)), decreasing = TRUE))
wordcloud(rownames(freq), freq[,1], max.words = 400, colors = brewer.pal(1, "Dark2"))

freq = data.frame(sort(colSums(as.matrix(trainingTokensMatrix)), decreasing = TRUE))
wordcloud(rownames(freq), freq[,1], max.words = 100, colors = brewer.pal(1, "Dark2"))

freq = data.frame(sort(colSums(as.matrix(testsTokensMatrix)), decreasing = TRUE))
wordcloud(rownames(freq), freq[,1], max.words = 100, colors = brewer.pal(1, "Dark2"))

# --------------------------------------------------------------------------------------------------------------------------------

#                                                            ***** TF-IDF *****
# TF(t, d): Proporción del recuento del término t en un documento d
# IDF: Frecuencia inversa del documento, que consiste en la proporción de la cantidad de documentos en la cual
#	   el término t se encuentra

# Luego, realizamos el TF-IDF sobre todo el corpus de datos


# Term Frequency (TF)
termFrequency <- function(row){
  row / sum(row)
}

# Inverse Document Frequency (IDF)
invDocFreq <- function(col){
  corpusSize <- length(col)
  docCount <- length(which(col > 0))

  log10(corpusSize / docCount)
}

# Función para calcular el TF-IDF.
tf_idf <- function(tf, idf) {
  tf * idf
}


dataTokensTF <- apply(dataTokensMatrix, 1, termFrequency)
dim(dataTokensTF)
View(dataTokensTF[1:100, 1:20])


# Segundo, se calcula el vector IDF que se usará para el training y para el test
dataTokensIDF <- apply(dataTokensMatrix, 2, invDocFreq)
str(dataTokensIDF)
View(dataTokensIDF)


# Por último, se calcula el TF-IDF para el corpus del training
dataTokensTFIDF <-  apply(dataTokensTF, 2, tf_idf, idf = dataTokensIDF)
dim(dataTokensTFIDF)
View(dataTokensTFIDF[1:25, 1:25])


# Se transpone la matriz
dataTokensTFIDF <- t(dataTokensTFIDF)
dim(dataTokensTFIDF)
View(dataTokensTFIDF[1:25, 1:25])


# Se verifican casos incompletos.
incomplete.cases <- which(!complete.cases(dataTokensTFIDF))
data$description[incomplete.cases]


# Se reparan casos incompletos
dataTokensTFIDF[incomplete.cases,] <- rep(0.0, ncol(dataTokensTFIDF))
dim(dataTokensTFIDF)
sum(which(!complete.cases(dataTokensTFIDF)))


# Se realiza una limpieza de los datos usando el mismo proceso anterior y se agrega el label
dataTokensTFIDF_df <- cbind(label = data$label, data.frame(dataTokensTFIDF))
names(dataTokensTFIDF_df) <- make.names(names(dataTokensTFIDF_df))

# --------------------------------------------------------------------------------------------------------------------------------

#                                                            ***** DATA SPLIT *****

# Ahora, se procede a dividir la información en los conjuntos de datos de entrenamiento de pruebas (training y tests), con una
# proporción respectiva del 80% y 20%. Este método es conocido como Cross-Validation o Validación Cruzada

# Con set.seed() nos aseguramos de que se mantenga el generador aleatorio para todos los conjuntos de datos
set.seed(3298)

# Asignamos la proporción para los conjuntos de datos
indices <- createDataPartition(dataTokensTFIDF_df$label, times = 1, p = 0.8, list = FALSE)

dataCV <- dataTokensTFIDF_df[indices,]
dataTests <- dataTokensTFIDF_df[-indices,]

prop.table(table(dataCV$label))
prop.table(table(dataTests$label))

# --------------------------------------------------------------------------------------------------------------------------------

#                                                           ***** CLASSIFICATION *****

# SVM Radial

# Cross-Validation
# Con 10 iteraciones o folds, para repetirlas 3 veces
foldscv <- createMultiFolds(dataCV$label, k = 10, times = 3)

control1 <- trainControl(method = "repeatedcv", number = 10, repeats = 3, index = foldscv)

# Tiempo de ejecución del código
startTime <- Sys.time()

# Se crea un cluster con 1 núcleo lógico (debido a la capacidad de mi computador)
cl <- makeCluster(1, type = "SOCK")
registerDoSNOW(cl)

# Para el modelo clasificación, se utilzia primero el SVM Radial
set.seed(3298)
svmRadial <- train(label ~ ., data = dataCV, method = "svmRadial", trControl = control1, tuneLength = 7)

# El procesamiento termina, se detiene el cluster
stopCluster(cl)

# Tiempo total de la ejecución
totalTime <- Sys.time() - startTime
totalTime

# Se visualizan la clasificación en el conjunto de entrenamiento
svmRadial
plot(svmRadial)

# Prediciendo los resultados para el dataset de pruebas
testsPredSVMr <- predict(svmRadial, newdata = dataTests)
testsPredSVMr

# Matriz de confusión, exactitud, precisión (valor positivo predicho) y recall (sensibilidad) de la clasificación
confusionMatrix(testsPredSVMr, dataTests$label)

# Decision Trees

foldscv <- createMultiFolds(dataCV$label, k = 10, times = 3)

control1 <- trainControl(method = "repeatedcv", number = 10, repeats = 3, index = foldscv)
startTime <- Sys.time()

cl <- makeCluster(1, type = "SOCK")
registerDoSNOW(cl)

set.seed(3298)
decisionTrees  <- train(label ~ . , data = dataCV, method = "rpart", trControl = control1, tuneLength = 7)

stopCluster(cl)

totalTime <- Sys.time() - startTime
totalTime

decisionTrees
plot(decisionTrees)

# Prediciendo los resultados para el dataset de pruebas
testsPredTrees <- predict(decisionTrees, newdata = dataTests)
testsPredTrees

# Matriz de confusión, exactitud, precisión (valor positivo predicho) y recall (sensibilidad) de la clasificación
confusionMatrix(testsPredTrees, dataTests$label)

