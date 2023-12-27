import java.util.ArrayList;
import java.util.Scanner;
import java.util.PriorityQueue;
//Gabriel Zohrob, 300309391
public class KNN {
		private int k; //value of k
		private ArrayList<LabelledPoint> pointsList;
	
		// Constructs the instance with a PointSet
		public KNN (PointSet d) {
			this.pointsList = d.getPointsList();
		}
		
		// set the value of k 
		public void setK(int k){
			this.k= k;	
		}
		
		//Method to return the k nearest neighbors of a query point
		public ArrayList<LabelledPoint> findNN(LabelledPoint q){
						
			for(LabelledPoint p : pointsList) {
				p.setKey(p.distanceTo(q));
			}
			
			PriorityQueue pq = new PriorityQueue<LabelledPoint>(pointsList);
			
			ArrayList<LabelledPoint> neighbors = new ArrayList<>();
			for (int i=0; i<k; i++)
				neighbors.add((LabelledPoint)pq.poll());
			
			return neighbors;
		}

		
		
		public static void main(String args[]) {
			
			// read the dataset and the query vectors
			PointSet queryPts = new PointSet(PointSet.read_ANN_SIFT("siftsmall_query.fvecs"));
			PointSet pointSet = new PointSet(PointSet.read_ANN_SIFT("siftsmall_base.fvecs"));
		
//			System.out.println("Query set: "+queryPts.getPointsList().size());
//			System.out.println("Point set: "+pointSet.getPointsList().size());

	        KNN knn = new KNN(pointSet);
	        knn.setK(10);
			
			ArrayList<LabelledPoint> neighbors;
			
			// for all query points
			for(LabelledPoint q : queryPts.getPointsList()) {
				
				// find the kNN
				neighbors = knn.findNN(q);
			
			    // print results
				System.out.print(q.getLabel() + " : ");
				for(int i=0; i<neighbors.size()-1; i++) {
					System.out.print(neighbors.get(i).getLabel() + ", ");
				}
				System.out.println(neighbors.get(neighbors.size()-1).getLabel());
	        }
		}	        
}