#!/usr/bin/env wolframscript
(* ::Package:: *)

(* ::Section:: *)
(*LeNet and Digit Recognition*)


(* ::Text:: *)
(*A brief overview of the Wolfram Language neural net framework by showing how to train a net that takes an input image of a handwritten single-digit number and then predicts the number. The dataset we are training on is the classic MNIST dataset, and we will train a variant of LeNet. *)


(* ::Text:: *)
(*Obtain the MNIST dataset, which contains 60,000 training and 10,000 test images:*)


(* ::Input:: *)
(*trainingData=ResourceData["MNIST","TrainingData"];*)
(*testData=ResourceData["MNIST","TestData"];*)


(* ::Text:: *)
(*Display a few random examples from the training set:*)


(* ::Input:: *)
(*RandomSample[trainingData,5]*)


(* ::Text:: *)
(*Construct an untrained version of LeNet:*)


(* ::Input:: *)
(*lenet=*)
(*NetChain[*)
(*{ConvolutionLayer[64,{3,3}],*)
(*ElementwiseLayer[Ramp],*)
(*ConvolutionLayer[32,{3,3}],*)
(*ElementwiseLayer[Ramp],*)
(*FlattenLayer[],*)
(*LinearLayer[10],*)
(*SoftmaxLayer[]},*)
(*"Input"->NetEncoder[{"Image",{28,28},ColorSpace->"Grayscale"}],*)
(*"Output"->NetDecoder[{"Class",Range[0,9]}]*)
(*];*)


(* ::Text:: *)
(*It is extremely easy to train a network like LeNet from scratch. RefLink[NetTrain,paclet:ref/NetTrain] takes care of many details of the training process automatically, such as selecting an appropriate loss function, attaching encoders and decoders and choosing a batch size. Here is what it looks like.*)


(* ::Text:: *)
(*Number of parameters to train:*)


(* ::Input:: *)
(*NetInformation[lenet,"ArraysTotalElementCount"]*)


(* ::Text:: *)
(*Train LeNet from scratch:*)


(* ::Input:: *)
(*digitRecognition=*)
(*NetTrain[*)
(*lenet,*)
(*trainingData,*)
(*ValidationSet->testData,*)
(*TargetDevice->"GPU",*)
(*MaxTrainingRounds->3*)
(*]*)


(* ::Input:: *)
(*digitRecognition = NetModel["LeNet Trained on MNIST Data"];*)


(* ::Text:: *)
(*Note:  \[Bullet]   Mutual entropy was chosen as loss function automatically.*)
(* 	   \[Bullet]   SGD method and parameters were chosen automatically.*)
(* 	   \[Bullet]   Weight initialization was done automatically.*)


(* ::Text:: *)
(*Testing the trained network. *)


(* ::Input:: *)
(*test=RandomSample[testData,10][[All,1]]*)


(* ::Input:: *)
(*digitRecognition[test]*)


(* ::Text:: *)
(*A comprehensive classifier test.*)


(* ::Input:: *)
(*cm=ClassifierMeasurements[digitRecognition, testData]*)


(* ::Text:: *)
(*Accuracy and error rate.*)


(* ::Input:: *)
(*cm["Accuracy"]*)


(* ::Input:: *)
(*cm["Error"]*)


(* ::Text:: *)
(*Confusion matrix.*)


(* ::Input:: *)
(*cm["ConfusionMatrixPlot"]*)
