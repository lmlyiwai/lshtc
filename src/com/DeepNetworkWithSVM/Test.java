package com.DeepNetworkWithSVM;

import java.io.File;
import java.io.IOException;
import java.util.Map;

import com.IMCLEF.ProcessIMCLEF;
import com.fileInputOutput.InvalidInputDataException;
import com.fileInputOutput.Parameter;
import com.fileInputOutput.Problem;
import com.rssvm.Measures;
import com.structure.Structure;

public class Test {

	@org.junit.Test
	public void test() throws IOException, InvalidInputDataException {
		String filename = "F:\\DataSets\\imclef07\\imclef07a\\imclef07a\\imclef07a_train.hf";
		String trainfile = "F:\\DataSets\\imclef07\\imclef07a\\imclef07a\\imclef07a_train";
		String testfile = "F:\\DataSets\\imclef07\\imclef07a\\imclef07a\\imclef07a_test";
		Map<String, Integer> map = ProcessIMCLEF.readStructure(filename);
		int[][] pc = ProcessIMCLEF.edge(filename, map);
					
		Structure tree = new Structure(map.size());
		for(int i = 0; i < pc.length; i++) {
			tree.addChild(pc[i][0], pc[i][1]);
		}
				
		int[] leaves = tree.getLeaves();
		
		Problem train = Problem.readProblem(new File(trainfile), 1, map);
		train.y = ProcessIMCLEF.pathToLeaf(train.y, leaves);
		
		Problem test = Problem.readProblem(new File(testfile), 1, map);
		test.y = ProcessIMCLEF.pathToLeaf(test.y, leaves);
		
		double[] cs = new double[15];
		for (int i = 0; i < cs.length; i++) {
			cs[i] = Math.pow(2, i-7);
		}
		
		String basepath = "F:\\program\\";
		for (int i = 0; i < cs.length; i++) {
			for (int j = 0; j < cs.length; j++) {
				Parameter p1 = new Parameter(cs[i], 1000, 0.001);
				Parameter p2 = new Parameter(cs[j], 1000, 0.001);
				Parameter[] params = {p1, p2};
				for (int k = 0; k < params.length; k++) {
					System.out.print(params[k].getC() + " ");
				}
				System.out.println();
				String path = basepath + cs[i] + "_" + cs[j];
				File file = new File(path);
				file.mkdir();
				long startTime = System.currentTimeMillis();
				train(train, test, params, path);
				long endTime = System.currentTimeMillis();
				System.out.println("Totle Time " + (endTime - startTime));
			}
		}
	}

	private static void train(Problem train, Problem test, Parameter[] params, String outputFileBase) throws IOException {
		TrainDeepSVM tds = new TrainDeepSVM(train, params, params.length);
		tds.train(outputFileBase);		
		int[][] predictLabels = tds.predict(test.x, outputFileBase);
		for (int col = 0; col < predictLabels[0].length; col++) {
			int[][] colPredictLabels = getColumnOfMatrix(predictLabels, col);
			double microf1 = Measures.microf1(tds.getUlabels(), test.y, colPredictLabels);
			double macrof1 = Measures.macrof1(tds.getUlabels(),  test.y,  colPredictLabels);
			double accuracy = Measures.averageSymLoss(test.y,  colPredictLabels);
			System.out.println("Layer " + col + ", MiF1 = " + microf1 + ", MaF1 = " + macrof1 + ", Accuracy = " + (1 - accuracy / 2));
		}
		
	}
	
	public static int[][] getColumnOfMatrix(int[][] matrix, int col) {
		if (matrix == null || col > matrix[0].length) {
			return null;
		}
		
		int[][] colOfMatrix = new int[matrix.length][1];
		for (int row = 0; row < matrix.length; row++) {
			colOfMatrix[row][0] = matrix[row][col];
		}
		return colOfMatrix;
	}
}
