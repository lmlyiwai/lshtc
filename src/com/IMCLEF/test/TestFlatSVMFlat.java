package com.IMCLEF.test;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.OutputStreamWriter;
import java.util.Map;

import org.junit.Test;

import com.IMCLEF.ProcessIMCLEF;
import com.fileInputOutput.InvalidInputDataException;
import com.fileInputOutput.Parameter;
import com.fileInputOutput.Problem;
import com.rssvm.OneVsAllMultilabel;
import com.sparseVector.DataPoint;
import com.structure.Structure;

public class TestFlatSVMFlat {

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
		Parameter param = new Parameter(2, 2000, 0.001);
		
		Problem prob = Problem.readProblem(new File(trainfile), 1, map);
		prob.y = ProcessIMCLEF.pathToLeaf(prob.y, leaves);
		
		
		OneVsAllMultilabel rs = new OneVsAllMultilabel(prob, param);
		DataPoint[][] w = rs.train(prob, param);
		
		Problem testprob = Problem.readProblem(new File(testfile), 1, map);
		testprob.y = ProcessIMCLEF.pathToLeaf(testprob.y, leaves);
		
		double[][] trainpv = rs.predictValues(w, prob.x);
//		rs.scale(trainpv);
		
		double[][] testpv = rs.predictValues(w, testprob.x);
//		rs.scale(testpv);
		
		String trainValue = "C:\\Users\\Administrator\\Desktop\\CLEF\\trainValue.txt";
		String testValue = "C:\\Users\\Administrator\\Desktop\\CLEF\\testValue.txt";
		writePredictValueLabelToFile(trainValue, trainpv, prob.y);
		writePredictValueLabelToFile(testValue, testpv, testprob.y);
		System.out.println("Written Done.");
	}

	public static void writePredictValueLabelToFile(String filename, double[][] pv, int[][] labels) throws IOException {
		BufferedWriter out = new BufferedWriter(
				new OutputStreamWriter(new FileOutputStream(filename)));
		String line = null;
		for(int i = 0; i < pv.length; i++) {
			line = new String();
			for(int j = 0; j < pv[i].length; j++) {
				line += pv[i][j] + " ";
			}
			line += labels[i][0] + "\n";
			out.write(line);
		}
		out.close();
	}
}
