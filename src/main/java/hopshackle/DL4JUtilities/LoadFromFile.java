package hopshackle.DL4JUtilities;

import java.io.*;
import java.util.*;
import java.util.stream.Collectors;

public class LoadFromFile {

    public static void main(String[] args) {
        try {
            HopshackleNN reloaded = HopshackleNN.createFromStream(new FileInputStream("C://simulation//hanabi//ConvertedDL4JModel.params"));

            List<List<Double>> X = new ArrayList<>();
            List<Double> Y = new ArrayList<>();
            FileReader reader = new FileReader("C:/simulation/hanabi/PowerPlantData.txt");
            BufferedReader bufferedReader = new BufferedReader(reader);
            do {
                String next = bufferedReader.readLine();
                if (next == null) break;
                String[] d = next.split("\\t");
                List<Double> data = Arrays.stream(d).map(Double::valueOf).collect(Collectors.toList());
                Y.add(data.get(0));
                data.remove(0);
                X.add(data);
            } while (true);

            // process file and output actual versus expected values

            for (int i = 0; i < X.size(); i++) {
                double[] temp = X.get(i).stream().mapToDouble(x -> x).toArray();
                double estY = reloaded.process(temp)[0];
                System.out.println(String.format("Line %3d\tY: %.2f\tY': %.2f", i + 1, Y.get(i), estY));
            }


        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
