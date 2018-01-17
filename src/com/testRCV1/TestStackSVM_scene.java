package com.testRCV1;

import java.io.File;
import java.io.IOException;

import org.junit.Test;

import com.fileInputOutput.InvalidInputDataException;
import com.fileInputOutput.Parameter;
import com.fileInputOutput.Problem;
import com.rssvm.Measures;
import com.rssvm.StackSVM;

public class TestStackSVM_scene {

	@Test
	public void test() throws IOException, InvalidInputDataException {
		String trainfile = "F:\\DataSets\\scene\\scene-train.svm";
		String testfile = "F:\\DataSets\\scene\\scene-test.svm";
		
		Problem train = Problem.readProblem(new File(trainfile), 1);
		Problem test = Problem.readProblem(new File(testfile), 1);
		Parameter param1 = new Parameter(0.3, 1000, 0.001);
		Parameter param2 = new Parameter(5, 1000, 0.001);
		StackSVM ss = new StackSVM(train, param1);
		
		ss.revisedTrain(train, param1, param2, null);
		
		int[][] pl = ss.predict(test.x, null);
		double microf1 = Measures.microf1(ss.getUniqueLabels(), test.y, pl);
		double macrof1 = Measures.macrof1(ss.getUniqueLabels(), test.y, pl);
		double hammingloss = Measures.averageSymLoss(test.y, pl);
		
		System.out.println("Micro-F1 = " + microf1 + ", Macro-F1 = " + macrof1 + 
				", Hamming Loss = " + hammingloss);
	}

}
