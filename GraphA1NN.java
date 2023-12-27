/* ---------------------------------------------------------------------------------
The GraphA1NN class is the starting class for the graph-based ANN search

(c) Robert Laganiere, CSI2510 2023
------------------------------------------------------------------------------------*/
//Gabriel Zohrob, 300309391
import java.util.List;
import java.util.PriorityQueue;
import java.util.Random;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashSet;
import java.util.LinkedList;
import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.StringTokenizer;

class GraphA1NN {

	UndirectedGraph<LabelledPoint> annGraph;
	private PointSet dataset;
	private int capacityS;
	private static int k;
	private static int s;
	private static GraphA1NN graph;
	private static PointSet querySet;

	// construct a graph from a file
	public GraphA1NN(String fvecs_filename) {

		annGraph = new UndirectedGraph<>();
		dataset = new PointSet(PointSet.read_ANN_SIFT(fvecs_filename));
	}

	// construct a graph from a dataset
	public GraphA1NN(PointSet set) {

		annGraph = new UndirectedGraph<>();
		this.dataset = set;
	}

	// build the graph.
	
	public void constructKNNGraph(int K) throws IOException, Exception {
		// build the graph using the provided adjacency list

    	ArrayList<List<Integer>> adjacency = readAdjacencyFile("knn.txt", dataset.getPointsList().size());

    	for (int i = 0; i < dataset.getPointsList().size(); i++) {
        	List<Integer> neighbors = adjacency.get(i);
        	LabelledPoint currentPoint = dataset.getPointsList().get(i);

			for (int j = 0; j < K; j++) {
				LabelledPoint neighborPoint = dataset.getPointsList().get(neighbors.get(j));
				annGraph.addEdge(currentPoint, neighborPoint);
			}
			
    }

		
		/*
		// this graph is useless but just to show how to build one
		for (int i = K; i < dataset.getPointsList().size(); i++) {
			for (int j = 1; j <= K; j++) {
				annGraph.addEdge(dataset.getPointsList().get(i),
						dataset.getPointsList().get(i - j));
			}
		}*/
	}

	public LabelledPoint find1NN(LabelledPoint queryPoint) {
		// Initialize a priority queue to hold vertices and their distances to the query point
		PriorityQueue<LabelledPointDistance> pq = new PriorityQueue<>(capacityS, new MyComparator());
	
		HashSet<LabelledPoint> checkedVertices = new HashSet<>();
	
		// Randomly select a starting vertex W from the graph
		LabelledPoint W = getRandomVertex();
		pq.offer(new LabelledPointDistance(W, W.distanceTo(queryPoint)));
	
		while (!pq.isEmpty()) {
			// Get the unchecked vertex C in the priority queue that has the smallest distance to the query point
			LabelledPointDistance current = pq.poll();
			LabelledPoint currentVertex = current.point;
	
			// Mark vertex C as checked
			checkedVertices.add(currentVertex);
	
			// If the current vertex is closer than any of its unprocessed neighbors, it's the nearest neighbor
			boolean isNearest = true;
			for (LabelledPoint neighbor : annGraph.getNeighbors(currentVertex)) {
				if (!checkedVertices.contains(neighbor)) {
					double dist = neighbor.distanceTo(queryPoint);
					pq.offer(new LabelledPointDistance(neighbor, dist));
					if (dist < current.distance) {
						isNearest = false;
					}
				}
			}
	
			if (isNearest) {
				return currentVertex; // Return the nearest neighbor
			}
		}
	
		// If the priority queue is empty and no neighbor was found, return null
		return null;
	}
	
	
	
	
	
	
	

	private static class LabelledPointDistance {
        LabelledPoint point;
        double distance;

        LabelledPointDistance(LabelledPoint point, double distance) {
            this.point = point;
            this.distance = distance;
        }
    }

	private Random random = new Random();

	private LabelledPoint getRandomVertex() {
		int randomIndex = random.nextInt(dataset.getPointsList().size());
		return dataset.getPointsList().get(randomIndex);
	}

	private class MyComparator implements Comparator<LabelledPointDistance> {

		public int compare(LabelledPointDistance p1, LabelledPointDistance p2) {
			Double key1 = p1.distance;
			Double key2 = p2.distance;
			return key1.compareTo(key2);
		}
	}



	public static ArrayList<List<Integer>> readAdjacencyFile(String fileName, int numberOfVertices)
			throws Exception, IOException {
		ArrayList<List<Integer>> adjacency = new ArrayList<List<Integer>>(numberOfVertices);
		for (int i = 0; i < numberOfVertices; i++)
			adjacency.add(new LinkedList<>()); //associate a new linkedlist to every vertex

		// read the file line by line
		String line;
		BufferedReader flightFile = new BufferedReader(new FileReader(fileName));

		// each line contains the vertex number followed
		// by the adjacency list
		while ((line = flightFile.readLine()) != null) {
			StringTokenizer st = new StringTokenizer(line, ":,");
			int vertex = Integer.parseInt(st.nextToken().trim());
			while (st.hasMoreTokens()) {
				adjacency.get(vertex).add(Integer.parseInt(st.nextToken().trim()));
			}
		}

		return adjacency;
	}

	public int size() {
		return annGraph.size();
	}

	public void setS(int S) {
		this.capacityS = S;
	}

	public static void main(String[] args) throws IOException, Exception {
		if (args.length != 4) {
			throw new IllegalArgumentException("Correct usage: java GraphA1NN <dataset> <query> <k> <s>");
		}
	
		String dataPath = args[0];
		String queryPath = args[1];
		int knn = Integer.parseInt(args[2]);
		int scale = Integer.parseInt(args[3]);
	
		GraphA1NN knnGraph = new GraphA1NN(dataPath);
		knnGraph.setS(scale);
		knnGraph.constructKNNGraph(knn);
	
		PointSet dataSet = new PointSet(PointSet.read_ANN_SIFT(dataPath));
		PointSet querySet = new PointSet(PointSet.read_ANN_SIFT(queryPath));
	
		int querySize = querySet.getPointsList().size();
		int dataSize = dataSet.getPointsList().size();  
	
		long totalTime = 0;
		BufferedWriter resultWriter = new BufferedWriter(new FileWriter("find1NN_" + knn + "_" + querySize + "_" + dataSize + ".txt"));
	
		for (int i = 0; i < querySize; i++) {
			LabelledPoint queryPoint = querySet.getPointsList().get(i);
			long startTime = System.currentTimeMillis();
			LabelledPoint nearestPoint = knnGraph.find1NN(queryPoint);
			long endTime = System.currentTimeMillis();
			totalTime += endTime - startTime;
			resultWriter.write(queryPoint.getLabel() + ": " + nearestPoint.getLabel() + "\n");
		}
		resultWriter.close();
	
		System.out.println("Execution time: " + totalTime + " milliseconds");
	
		ArrayList<List<Integer>> referenceAnswers = GraphA1NN.readAdjacencyFile("knn_3_10_100_10000.txt", 100);
		ArrayList<List<Integer>> generatedAnswers = GraphA1NN.readAdjacencyFile("find1NN_" + knn + "_" + querySize + "_" + dataSize + ".txt", 100);
	
		int correctMatches = 0;
		for (int i = 0; i < referenceAnswers.size(); i++) {
			if (!generatedAnswers.isEmpty() && generatedAnswers.get(i).size() > 0) {
				int generatedLabel = generatedAnswers.get(i).get(0);
				for (int refLabel : referenceAnswers.get(i)) {
					if (generatedLabel == refLabel) {
						correctMatches++;
						break;
					}
				}
			}
		}
	
		double accuracyPercentage = (double) correctMatches / referenceAnswers.size() * 100;
		System.out.println("Value of k: " + knn);
		System.out.println("Value of s: " + scale);
		System.out.println("Accuracy percentage: " + accuracyPercentage);
	}
	
	
}
