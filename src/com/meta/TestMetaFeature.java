package com.meta;

import java.io.File;
import java.io.IOException;

import org.junit.Test;

import com.fileInputOutput.InvalidInputDataException;
import com.fileInputOutput.Problem;
import com.sparseVector.DataPoint;

public class TestMetaFeature {

	@Test
	public void test() throws IOException, InvalidInputDataException {
		String trainfile = "test.txt";
		String testfile = "train.txt";
		
		Problem train = Problem.readProblem(new File(trainfile), -1);
		Problem test = Problem.readProblem(new File(testfile), -1);
		MetaFeature mf = new MetaFeature(train);
		double[][] tf = mf.transTrainSet(train, 1);
		double[][] testf = mf.transTestSet(train, test, 1);
		
		for(int i = 0; i < tf.length; i++) {
			for(int j = 0; j < tf[i].length; j++) {
				System.out.print(tf[i][j] + " ");
			}
			System.out.println();
		}
		
		DataPoint[][] x = mf.trans(tf);
		for(int i = 0; i < x.length; i++) {
			for(int j = 0; j < x[i].length; j++) {
				System.out.print(x[i][j].index + ":" + x[i][j].value + " ");
			}
			System.out.println();
		}
		
		System.out.println(tf.length + "--" + tf[0].length);
	}

}
