/* ---------------------------------------------------------------------------------
The PointSet class that contains an ArraList of LabelledPoint instances.
It also reads fvecs files from corpus-texmex.irisa.fr


(c) Robert Laganiere, CSI2510 2023
------------------------------------------------------------------------------------*/
//Gabriel Zohrob, 300309391
import java.io.*;
import java.util.ArrayList;

class PointSet {
    private ArrayList<LabelledPoint> pointsList = new ArrayList<>();
    public PointSet(){  
        
    }

    public PointSet(ArrayList<LabelledPoint> pointsList){
       this.pointsList = pointsList;
    }

    public static ArrayList<LabelledPoint> read_ANN_SIFT(String filename) {
        ArrayList<LabelledPoint> pointSet = new ArrayList<>();
        try {
            FileInputStream fis = new FileInputStream(filename);
            BufferedInputStream bis = new BufferedInputStream(fis);
            DataInputStream dis = new DataInputStream(bis);

            int d = Integer.reverseBytes(dis.readInt());
            int vecSizeOf = 1 * 4 + d * 4;
            long fileLength = new File(filename).length();
            int bMax = (int) (fileLength / vecSizeOf);
            int a = 1;
            int b = bMax;

            if( a >= 1 && b>bMax){
                b =bMax;
            }

            if (b == 0 || b < a) {
                dis.close();
				throw new IOException("Error! Invalid file format...");
            }

            int n = b - a + 1;
            
            dis.skipBytes((a - 1) * vecSizeOf);
            for (int i = 0; i < n; i++) {
                Float[] vector = new Float[d];
                for (int j = 0; j < d; j++) {
                    int floatAsInt = Integer.reverseBytes(dis.readInt());
                    vector[j] = Float.intBitsToFloat(floatAsInt);
                }
                dis.skipBytes(4);
                pointSet.add(new LabelledPoint(vector,i));
            }

            dis.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
        return pointSet;
    }

    public ArrayList<LabelledPoint> getPointsList() {
        return pointsList;
    }

    public void setPointsList (ArrayList<LabelledPoint> pointsList){
        this.pointsList = pointsList;
    }

    public static void main(String[] args) {

		// read the dataset and the query vectors
        PointSet queryPts = new PointSet(PointSet.read_ANN_SIFT("siftsmall_query.fvecs"));
        PointSet pointSet = new PointSet(PointSet.read_ANN_SIFT("siftsmall_base.fvecs"));
		
        System.out.println("Query set: "+queryPts.getPointsList().size());
        System.out.println("Point set: "+pointSet.getPointsList().size());

        double avgtime=0.0;
		
		// for all query points
		for (int j=0; j<queryPts.getPointsList().size(); j++) {
		
		    // finf the 1NN
			long startTime = System.currentTimeMillis();
			
			double distmin= 1000000000.f;
			int minIndex=-1;
		
			for (int i=0; i<pointSet.getPointsList().size(); i++) {
			
			    // compute distance from query to dataset vector
				double dist= pointSet.getPointsList().get(i).distanceTo(queryPts.getPointsList().get(j));
				if (dist < distmin) { // we found a nearer vector
					distmin= dist;
					minIndex= i;
				}
			}
			
			long endTime = System.currentTimeMillis();
			avgtime+= (double)(endTime - startTime); // time to execute a query
		
		    // never include i/o statements inside blocks you are timing execution time!!
			System.out.println(j + " : " + minIndex);
		}
		
		// print the average time to execute a query
		avgtime/= (double)(queryPts.getPointsList().size());
        System.out.println("Average execution time: " + avgtime + " milliseconds");
	}
}

