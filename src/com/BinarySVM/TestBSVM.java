package com.BinarySVM;

import java.io.File;
import java.io.IOException;

import org.junit.Test;

import com.fileInputOutput.InvalidInputDataException;
import com.fileInputOutput.Parameter;
import com.fileInputOutput.Problem;
import com.sparseVector.DataPoint;
import com.sparseVector.SparseVector;
import com.structure.Structure;
import com.tools.FileIO;

public class TestBSVM {

	@Test
	public void test() throws IOException, InvalidInputDataException {
		String filename = "test.txt";
		filename = "F:\\DataSets\\arti\\data.txt";
		String outputfile = "F:\\DataSets\\arti\\samples.txt";
		FileIO.trans(filename, outputfile);
		
		Structure tree = new Structure(7);
		tree.addChild(0, 1);
		tree.addChild(0, 2);
		tree.addChild(1, 3);
		tree.addChild(1, 4);
		tree.addChild(2, 5);
		tree.addChild(2, 6);
		
		Parameter param = new Parameter(1, 1000, 0.001);
		
		Problem train = Problem.readProblem(new File(filename), 1);
		BinarySVM bs = new BinarySVM(tree);
		bs.getUlabels(train.y);
//		DataPoint[][] w = bs.newTrain(train, param);
		DataPoint[][] w = bs.train(train, param);
		
		int[][] pre = bs.predict(w, train.x);
		for(int i = 0; i < pre.length; i++) {
			for(int j = 0; j < pre[i].length; j++) {
				System.out.print(pre[i][j] + " ");
			}
			System.out.println();
		}
		
		for(int i = 0; i < w.length; i++) {
			SparseVector.showVector(w[i]);
		}
	}

}
