package com.flatSvm;

import java.io.File;
import java.io.IOException;

import org.junit.Test;

import com.fileInputOutput.InvalidInputDataException;
import com.fileInputOutput.Parameter;
import com.fileInputOutput.Problem;
import com.meta.MetaFeature;
import com.rssvm.Measures;

public class TestScene {

	@Test
	public void test() throws IOException, InvalidInputDataException {
		String trainfile = "F:\\java\\RecursiveRegularizationSVM_1\\CLEF_train.txt";
		String testfile = "F:\\java\\RecursiveRegularizationSVM_1\\CLEF_test.txt";
		
		String scene_weight_file = "clef_weight_file.txt";
		
//		String trainfile = "F:\\DataSets\\yeast\\yeast_train.svm";
//		String testfile = "F:\\DataSets\\yeast\\yeast_test.svm";
		
//		String trainfile = "test.txt";
//		String testfile = "test.txt";
		
		
		Problem train = Problem.readProblem(new File(trainfile), 1);
		Problem test = Problem.readProblem(new File(testfile), 1);
		
		Parameter param = new Parameter(1, 1000, 0.001);	
		FlatSVM fs = new FlatSVM(train, param);
		param.setC(1);
		fs.train(train, param, scene_weight_file);
		double[][] tpv = fs.predictValues(train.x, scene_weight_file);
		double[][] testpv = fs.predictValues(test.x, scene_weight_file);
		
		int[] tpl = fs.predictMax(tpv);
		double tacc = fs.accuracy(train.y, tpl);
		
		int[] pl = fs.predictMax(testpv);
		double acc = fs.accuracy(test.y, pl);
		System.out.println("Test Accuracy = " + acc + ", Train Accuracy = " + tacc);
		
//		double[][] t = fs.getTPbpu(tpv, train.y, testpv, test.y, 10);
		double[][] w = fs.ann(tpv, train.y, 0.0005, 1000, testpv, test.y);
	}

}
