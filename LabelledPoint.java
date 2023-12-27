/* ---------------------------------------------------------------------------------
A vector of Floats with its associated int label
It includes also a key that can be used to order LabelledPoint instances
This new version also includes an integer key
and a boolean flag

(c) Robert Laganiere, CSI2510 2023
------------------------------------------------------------------------------------*/
//Gabriel Zohrob, 300309391
class LabelledPoint implements Comparable<LabelledPoint> {
    private Float[] vector;   // the vector
	private int label;        // the id (should be unique)
	private double key;       // the key used for ordering
	private int ikey;         // an additional integer key
	private boolean checked;  // a boolean flag

    public LabelledPoint(Float[] vector, int label) {
        this.vector = vector;
		this.label= label;
		this.key= 0.0;
		this.ikey= -1;
		this.checked= false;
    }
	
	// gets the label
	public int getLabel() {
	
	    return label;
	}
	
	// gets the vector (an array of floats)
	public Float[] getVector(){
	
	    return vector;
	}
	
	// gets the key value
	public double getKey() {
		return key;
	}
	
	// set the key value
	public void setKey(double k) {
		key= k;
	}

	// gets the int key value
	public double getIKey() {
		return ikey;
	}
	
	// set the int key value
	public void setIKey(int k) {
		ikey= k;
	}
	
	// check flag 
	public boolean isChecked() {
		return checked;
	}	

	// set checked flag to true
	public void checked() {
		checked= true;
	}	

	// set checked flag to false
	public void unchecked() {
		checked= false;
	}	

    // gets the length (dimension) of the vector	
	public int getLength() {
	    return vector.length;	
	}

    // computes the Euclidean distance between two vectors
    public double distanceTo(LabelledPoint other) {
        double sum = 0;
        for (int i = 0; i < vector.length; i++) {
            float diff = vector[i] - other.vector[i];
            sum += diff * diff;
        }
        return Math.sqrt(sum);
    }

    // compares two LabelledPoint instances	
	public int compareTo(LabelledPoint o) {
		return Double.compare(this.key, o.key);
	}

    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder("(");
        for (int i=0; i<4; i++) {
            sb.append(vector[i]).append(", ");
        }
        sb.append("..., ");
        for (int i=vector.length-2; i<vector.length; i++) {
            sb.append(vector[i]).append(", ");
        }
        sb.setLength(sb.length() - 2); 
        sb.append(")");
        return sb.toString();
    }
}
