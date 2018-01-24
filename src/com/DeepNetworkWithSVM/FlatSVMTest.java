package com.DeepNetworkWithSVM;

import static org.junit.Assert.*;

import java.io.File;
import java.io.IOException;
import java.util.Map;

import org.junit.Test;

import com.IMCLEF.ProcessIMCLEF;
import com.fileInputOutput.InvalidInputDataException;
import com.fileInputOutput.Parameter;
import com.fileInputOutput.Problem;
import com.flatSvm.FlatSVM;
import com.rssvm.Measures;
import com.sparseVector.DataPoint;
import com.structure.Structure;

public class FlatSVMTest {

	@Test
	public void test() throws IOException, InvalidInputDataException {
		String filename = "F:\\DataSets\\imclef07\\imclef07a\\imclef07a\\imclef07a_train.hf";
		String trainfile = "F:\\DataSets\\imclef07\\imclef07a\\imclef07a\\imclef07a_train";
		String testfile = "F:\\DataSets\\imclef07\\imclef07a\\imclef07a\\imclef07a_test";
		Map<String, Integer> map = ProcessIMCLEF.readStructure(filename);
		int[][] pc = ProcessIMCLEF.edge(filename, map);
				
		Map<Integer, String> idTostr = ProcessIMCLEF.reverseMap(map);
		
		Structure tree = new Structure(map.size());
		for(int i = 0; i < pc.length; i++) {
			tree.addChild(pc[i][0], pc[i][1]);
		}
				
		int[] leaves = tree.getLeaves();
		Parameter param = new Parameter(1, 1000, 0.001);
		Problem train = Problem.readProblem(new File(trainfile), 1, map);
		train.y = ProcessIMCLEF.pathToLeaf(train.y, leaves);
		
		FlatSVM fs = new FlatSVM(train, param);
		fs.train(train, param);
		
		Problem test = Problem.readProblem(new File(testfile), 1, map);
		test.y = ProcessIMCLEF.pathToLeaf(test.y, leaves);
		int[] p = fs.predictMax(test.x);
		int[][] pre = transform(p);
		
		double microf1 = Measures.microf1(fs.getUlabels(), test.y, pre);
		double macrof1 = Measures.macrof1(fs.getUlabels(), test.y, pre);
		double hamLos = Measures.averageSymLoss(test.y, pre);
		System.out.println("MiF1 = " + microf1 + ", MaF1 = " + macrof1 + ", Accuracy = " + (1 - hamLos / 2));

	}
	
	private static int[][] transform(int[] y) {
		if (y == null) {
			return null;
		}
		int[][] result = new int[y.length][1];
		for (int i = 0; i < result.length; i++) {
			result[i][0] = y[i];
		}
		return result;
	}

}
