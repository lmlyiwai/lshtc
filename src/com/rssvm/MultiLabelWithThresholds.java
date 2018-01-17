package com.rssvm;

import java.util.Map;

import com.fileInputOutput.Parameter;
import com.fileInputOutput.Problem;
import com.sparseVector.DataPoint;
import com.sparseVector.SparseVector;
import com.threshold.Scutfbr;
import com.tools.Contain;
import com.tools.Statistics;

public class MultiLabelWithThresholds {
	private Problem 		prob;
	private Parameter		param;
	private int[] 			labels;
	private DataPoint[][] 	w;
	private double[][]     	b;
	
	
	public MultiLabelWithThresholds(Problem prob,  Parameter param) {
		this.prob = prob;
		this.param = param;
		this.labels = Statistics.getUniqueLabels(prob.y);
		this.w = new DataPoint[this.labels.length][];
		this.b = new double[this.labels.length][];
	}
	
	public void train() {
		int label;
		int j;
		int[] y;
		for(int i = 0; i < labels.length; i++) {
			label = labels[i];
System.out.print("label " + label);
			y = new int[prob.l];
			for(j = 0; j < prob.l; j++) {
				if(Contain.contain(prob.y[j], label)) {
					y[j] = 1;
				} else {
					y[j] = -1;
				}
			}
			
			Map<String, Object> map = Scutfbr.trainOneLabel(prob, y, param);
			w[i] = (DataPoint[]) map.get("weight");
			b[i] = (double[])map.get("b");
System.out.println(", b = " + b[i][0]);
		}
	}
	
	public int[][] predict(Problem testprob) {
		if(testprob == null) {
			return null;
		}
		
		int i, j;
		
		double[][] weight = new double[this.w.length][];
		for(i = 0; i < weight.length; i++) {
			weight[i] = SparseVector.sparseVectorToArray(this.w[i], this.prob.n);
		}

		DataPoint[] sample;
		int[][] pre = new int[testprob.l][];
		double[] temp;
		int counter;
		for(i = 0; i < testprob.l; i++) {
			sample = testprob.x[i];
			temp = new double[this.labels.length];
			counter = 0;
			for(j = 0; j < this.labels.length; j++) {
				temp[j] = SparseVector.innerProduct(weight[j], sample) + b[j][0];
				if(temp[j] > 0) {
					counter++;
				}
			}
			
			pre[i] = new int[counter];
			counter = 0;
			for(j = 0; j < temp.length; j++) {
				if(temp[j] > 0) {
					pre[i][counter++] = this.labels[j];
				}
			}
		}
		return pre;
	}

	public int[] getLabels() {
		return labels;
	}

	public void setLabels(int[] labels) {
		this.labels = labels;
	}

	public DataPoint[][] getW() {
		return w;
	}

	public void setW(DataPoint[][] w) {
		this.w = w;
	}

	public double[][] getB() {
		return b;
	}

	public void setB(double[][] b) {
		this.b = b;
	}
	
}
