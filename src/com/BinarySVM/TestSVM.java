package com.BinarySVM;

import java.io.File;
import java.io.IOException;

import javax.xml.crypto.Data;

import org.junit.Test;

import com.fileInputOutput.InvalidInputDataException;
import com.fileInputOutput.Parameter;
import com.fileInputOutput.Problem;
import com.rssvm.Linear;
import com.sparseVector.DataPoint;
import com.sparseVector.SparseVector;

public class TestSVM {

	@Test
	public void test() throws IOException, InvalidInputDataException {
		String outputfile = "F:\\DataSets\\arti\\samples.txt";
		
		Problem prob = Problem.readProblem(new File(outputfile), 1);
		Parameter param = new Parameter(1, 1000, 0.001);
		int[] labels = getLabels(prob.y);
		double[] mar = cost(labels);
		
//		DataPoint[] w = Linear.train(prob, labels, param, mar);
		double[] tloss = new double[1];
		DataPoint[] w = Linear.train(prob, labels, param, null, mar, tloss, null, 0);
		SparseVector.showVector(w);
	}
	
	public static int[] getLabels(int[][] y) {
		int[] labels = new int[y.length];
		for(int i = 0; i < y.length; i++) {
			labels[i] = y[i][0];
		}
		return labels;
	}
	
	public static double[] cost(int[] labels) {
		double[] cost = new double[labels.length];
		for(int i = 0; i < labels.length; i++) {
			if(labels[i] == 1) {
				cost[i] = 100;
			} else {
				cost[i] = 1;
			}
		}
		return cost;
	}
}
