package com.testRCV1;

import java.io.File;
import java.io.IOException;
import java.util.Map;

import org.junit.Test;

import com.fileInputOutput.InvalidInputDataException;
import com.fileInputOutput.Parameter;
import com.fileInputOutput.Problem;
import com.rssvm.Measures;
import com.rssvm.StackSVM;
import com.sparseVector.DataPoint;
import com.structure.Structure;
import com.tools.FileIO;
import com.tools.FileInputOutput;

public class TestSTtackSVM_RCV1 {

	@Test
	public void test() throws IOException, InvalidInputDataException {
		String filename = "./RCV1RCV2/rcv1.topics.hierorig.txt";
		String qrelfilename = "./RCV1RCV2/rcv1-v2.topics.qrels";
		
		String[][] result = FileInputOutput.readLabelPairs(filename);
		Map<String, Integer> map = FileInputOutput.getIDLabelPair(result);
		
		int[][] pc = FileInputOutput.labelPairs(result, map);
		Structure tree = new Structure(map.size());
		for(int i = 0; i < pc.length; i++) {
			tree.addChild(pc[i][0], pc[i][1]);
		}
		
		String trainFile = "./RCV1RCV2/vectors/lyrl2004_vectors_train.dat";
		String testfile0 = "./RCV1RCV2/vectors/lyrl2004_vectors_test_pt0.dat";
		String testfile1 = "./RCV1RCV2/vectors/lyrl2004_vectors_test_pt1.dat";
		String testfile2 = "./RCV1RCV2/vectors/lyrl2004_vectors_test_pt2.dat";
		String testfile3 = "./RCV1RCV2/vectors/lyrl2004_vectors_test_pt3.dat";
	
		Problem prob = FileIO.readProblem(new File(trainFile), 1);
	
		Map<Integer, int[]> iamap = FileInputOutput.getDocLabel(qrelfilename, map);
		prob.getFileLabels(iamap);
		
		Parameter param1 = new Parameter(0.5, 1000, 0.001);
		Parameter param2 = new Parameter(5, 1000, 0.001);
		StackSVM rs = new StackSVM(prob, param1);				
		
		rs.revisedTrain(prob, param1, param2, tree);
		
		int[][] pl = rs.predict(prob.x, tree);
		double microf1 = Measures.microf1(rs.getUniqueLabels(), prob.y, pl);
		double macrof1 = Measures.macrof1(rs.getUniqueLabels(), prob.y, pl);
		double hammingloss = Measures.averageSymLoss(prob.y, pl);
		System.out.println("Microf1 = " + microf1 + ", Macrof1 = " + macrof1 +
				", Hamming loss = " + hammingloss);
		
		Problem testprob0 = FileIO.readProblem(new File(testfile0), 1);	
		testprob0.getFileLabels(iamap);
		int[][] pre0 = rs.predict(testprob0.x, tree);
		
		Problem testprob1 = FileIO.readProblem(new File(testfile1), 1);
		testprob1.getFileLabels(iamap);
		int[][] pre1 = rs.predict(testprob1.x, tree);
		
		Problem testprob2 = FileIO.readProblem(new File(testfile2), 1);
		testprob2.getFileLabels(iamap);
		int[][] pre2 = rs.predict(testprob2.x, tree);
		
		Problem testprob3 = FileIO.readProblem(new File(testfile3), 1);
		testprob3.getFileLabels(iamap);
		int[][] pre3 = rs.predict(testprob3.x, tree);
	
		int[][] pre = new int[pre0.length + pre1.length + pre2.length + pre3.length][];
		int[][] testl = new int[pre0.length + pre1.length + pre2.length + pre3.length][];
		
		int counter = 0;
		for(int i = 0; i < pre0.length; i++) {
			pre[counter] = pre0[i];
			testl[counter] = testprob0.y[i];
			counter++;
		}
		
		for(int i = 0; i < pre1.length; i++) {
			pre[counter] = pre1[i];
			testl[counter] = testprob1.y[i];
			counter++;
		}
		
		for(int i = 0; i < pre2.length; i++) {
			pre[counter] = pre2[i];
			testl[counter] = testprob2.y[i];
			counter++;
		}
		
		for(int i = 0; i < pre3.length; i++) {
			pre[counter] = pre3[i];
			testl[counter] = testprob3.y[i];
			counter++;
		}

		double test_microf1 = Measures.microf1(rs.getUniqueLabels(), testl, pre);
		double test_macrof1 = Measures.macrof1(rs.getUniqueLabels(), testl, pre);
		double test_hamming = Measures.averageSymLoss(testl, pre);
		
		System.out.println("Micro-F1 = " + test_microf1 + ", Macro-F1 = " + test_macrof1 + 
				", Hamming Loss = " + test_hamming);
	}

}
