package com.consistance;

import java.io.File;
import java.io.IOException;

import org.junit.Test;

import com.fileInputOutput.InvalidInputDataException;
import com.fileInputOutput.Parameter;
import com.fileInputOutput.Problem;
import com.sparseVector.DataPoint;
import com.sparseVector.SparseVector;
import com.structure.Structure;

public class TestCOnsistentSVM {

	@Test
	public void test() throws IOException, InvalidInputDataException {
		Structure tree = new Structure(7);
		tree.addChild(0, 1);
		tree.addChild(0, 2);
		tree.addChild(1, 3);
		tree.addChild(1, 4);
		tree.addChild(2, 5);
		tree.addChild(2, 6);
		
		String trainfile = "test.txt";
		Problem train = Problem.readProblem(new File(trainfile), 1);
		
		Parameter param = new Parameter(10, 1, 1000, 0.001);
		
		ConsistentSVM cs = new ConsistentSVM(train, param, tree);
		cs.train(train, param);
		
		int[][] pl = cs.predict(train.x);
		for(int i = 0; i < pl.length; i++) {
			for(int j = 0; j < pl[i].length; j++) {
				System.out.print(pl[i][j] + " ");
			}
			System.out.println();
		}
		
		DataPoint[][] w = cs.getW();
		for(int i = 0; i < w.length; i++) {
			SparseVector.showVector(w[i]);
		}
		
		int[] label = cs.getLabels();
		for(int i = 0; i < label.length; i++) {
			System.out.println(label[i]);
		}
	}

}
