import java.util.Arrays;

public class BackpropagationNetworkTest {
    public static void main(String[] args) {
// Обучающие данные
        double[][] inputs = {
                {0,0,0,0,0,0,0,0,0,0},
                {0,0,0,0,0,0,0,0,0,1},
                {0,0,0,0,0,0,0,0,1,0},
                {0,0,0,0,0,0,0,0,1,1},
                {0,0,0,0,0,0,0,1,0,0},
                {0,0,0,0,0,0,0,1,0,1},
                {0,0,0,0,0,0,0,1,1,0},
                {0,0,0,0,0,0,0,1,1,1},
                {0,0,0,0,0,0,1,0,0,0},
                {0,0,0,0,0,0,1,0,0,1}
        };
        double[][] targets = {
                {0,0},
                {0,1},
                {0,1},
                {0,0},
                {0,1},
                {0,0},
                {0,0},
                {0,1},
                {0,1},
                {0,0}
        };
        BackpropagationNetwork network = new BackpropagationNetwork(10, 10, 2, inputs, targets, 0.1);
        network.train(1000); // обучение сети
        System.out.println(Arrays.toString(network.recognize(new double[]{0,0,0,0,0,0,0,0,0,0}))); // должно быть [0.0, 0.0]
        System.out.println(Arrays.toString(network.recognize(new double[]{0,0,0,0,0,0,0,0,0,1}))); // должно быть [0.0, 1.0]
        System.out.println(Arrays.toString(network.recognize(new double[]{0,0,0,0,0,0,0,1,0,0}))); // должно быть [0.0, 1.0]
        System.out.println(Arrays.toString(network.recognize(new double[]{0,0,0,0,0,0,0,1,0,1}))); // должно быть [0.0, 1.0]
        System.out.println(Arrays.toString(network.recognize(new double[]{0,0,0,0,0,0,0,1,1,0}))); // должно быть [0.0, 0.0]
        System.out.println(Arrays.toString(network.recognize(new double[]{0,0,0,0,0,0,0,1,1,1}))); // должно быть [0.0, 1.0]
        System.out.println(Arrays.toString(network.recognize(new double[]{0,0,0,0,0,0,1,0,0,0}))); // должно быть [0.0, 1.0]
        System.out.println(Arrays.toString(network.recognize(new double[]{0,0,0,0,0,0,1,0,0,1}))); // должно быть [0.0, 0.0]
        System.out.println(Arrays.toString(network.recognize(new double[]{0,0,0,0,0,0,1,0,1,0}))); // должно быть [0.0, 0.0]
        System.out.println(Arrays.toString(network.recognize(new double[]{0,0,0,0,0,0,1,0,1,1}))); // должно быть [0.0, 1.0]
        System.out.println(Arrays.toString(network.recognize(new double[]{0,0,0,0,0,0,1,1,0,0}))); // должно быть [0.0, 0.0]
    }

}
