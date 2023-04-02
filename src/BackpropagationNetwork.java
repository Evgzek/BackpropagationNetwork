import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;

public class BackpropagationNetwork {
    private final int numInput; // количество входных нейронов
    private final int numHidden; // количество скрытых нейронов
    private final int numOutput; // количество выходных нейронов
    private final double[][] inputs; // обучающие входные данные
    private final double[][] targets; // целевые выходные данные
    private double[][] hiddenWeights; // веса между входным и скрытым слоем
    private double[][] outputWeights; // веса между скрытым и выходным слоем
    private double[] hiddenBiases; // смещения скрытых нейронов
    private double[] outputBiases; // смещения выходных нейронов
    private final double learningRate; // скорость обучения

    public BackpropagationNetwork(int numInput, int numHidden, int numOutput, double[][] inputs, double[][] targets, double learningRate) {
        this.numInput = numInput;
        this.numHidden = numHidden;
        this.numOutput = numOutput;
        this.inputs = inputs;
        this.targets = targets;
        this.learningRate = learningRate;
        this.hiddenWeights = new double[numInput][numHidden];
        this.outputWeights = new double[numHidden][numOutput];
        this.hiddenBiases = new double[numHidden];
        this.outputBiases = new double[numOutput];
        initializeWeights(); // инициализация весов небольшими случайными значениями
    }

    // Инициализация весов
    private void initializeWeights() {
        Random random = new Random();
        for (int i = 0; i < numInput; i++) {
            for (int j = 0; j < numHidden; j++) {
                hiddenWeights[i][j] = random.nextDouble() - 0.5; // случайные значения в диапазоне [-0.5, 0.5]
            }
        }
        for (int i = 0; i < numHidden; i++) {
            for (int j = 0; j < numOutput; j++) {
                outputWeights[i][j] = random.nextDouble() - 0.5;
            }
        }
        for (int i = 0; i < numHidden; i++) {
            hiddenBiases[i] = random.nextDouble() - 0.5;
        }
        for (int i = 0; i < numOutput; i++) {
            outputBiases[i] = random.nextDouble() - 0.5;
        }
    }

    // Функция активации (сигмоида)
    private double sigmoid(double x) {
        return 1.0 / (1.0 + Math.exp(-x));
    }

    // Прямой проход
    private double[] forward(double[] input) {
        double[] hidden = new double[numHidden];
        double[] output = new double[numOutput];
        for (int i = 0; i < numHidden; i++) {
            double sum = 0.0;
            for (int j = 0; j < numInput; j++) {
                sum += input[j] * hiddenWeights[j][i];
            }
            hidden[i] = sigmoid(sum + hiddenBiases[i]);
        }
        for (int i = 0; i < numOutput; i++) {
            double sum = 0.0;
            for (int j = 0; j < numHidden; j++) {
                sum += hidden[j] * outputWeights[j][i];
            }
            output[i] = sigmoid(sum + outputBiases[i]);
        }
        return output;
    }

    // Обратный проход
    private void backward(double[] input, double[] output, double[] target) {
        double[] outputError = new double[numOutput];
        double[] hiddenError = new double[numHidden];
        for (int i = 0; i < numOutput; i++) {
            double error = output[i] * (1 - output[i]) * (target[i] - output[i]);
            outputError[i] = error;
            for (int j = 0; j < numHidden; j++) {
                outputWeights[j][i] += learningRate * hiddenError[j] * error;
            }
            outputBiases[i] += learningRate * error;
        }

        for (int i = 0; i < numHidden; i++) {
            double sum = 0.0;
            for (int j = 0; j < numOutput; j++) {
                sum += outputError[j] * outputWeights[i][j];
            }
            double error = hiddenError[i] * (1 - hiddenError[i]) * sum;
            hiddenError[i] = error;
            for (int j = 0; j < numInput; j++) {
                hiddenWeights[j][i] += learningRate * input[j] * error;
            }
            hiddenBiases[i] += learningRate * error;
        }
    }

    // Тренировка сети
    public void train(int epochs) {
        for (int i = 0; i < epochs; i++) {
            double error = 0.0;
            for (int j = 0; j < inputs.length; j++) {
                double[] output = forward(inputs[j]);
                backward(inputs[j], output, targets[j]);
                for (int k = 0; k < numOutput; k++) {
                    error += 0.5 * (targets[j][k] - output[k]) * (targets[j][k] - output[k]);
                }
            }
            System.out.println("Epoch " + (i+1) + " Error: " + error);
        }
    }

    // Распознавание
    public double[] recognize(double[] input) {
        return forward(input);
    }



}



