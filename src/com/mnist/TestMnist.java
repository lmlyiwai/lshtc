package com.mnist;

import java.io.File;
import java.io.IOException;

import org.junit.Test;

import com.fileInputOutput.InvalidInputDataException;
import com.fileInputOutput.Parameter;
import com.fileInputOutput.Problem;
import com.sparseVector.DataPoint;

public class TestMnist {

	@Test
	public void test() throws IOException, InvalidInputDataException {
		String trainfile = "F:\\java\\ProcessMNIST\\train.txt";
		String testfile = "F:\\java\\ProcessMNIST\\test.txt";
		
		Problem train = Problem.readProblem(new File(trainfile), 1);
		Problem test = Problem.readProblem(new File(testfile), 1);
		
		train = Mnist.filter(train, 3, 8);
		test = Mnist.filter(test, 3, 8);
		
		int[] k = {3,5,7,9,11,21,31,41,51,61,71,81};
		
		double[] c = new double[15];
		for(int i = 0; i < 15; i++) {
			c[i] = Math.pow(2, i - 7);
		}
		
		Parameter param = new Parameter(1, 1000, 0.001);
		for(int i = 0; i < c.length; i++) {
			for(int j = 0; j < k.length; j++) {
//				param.setC(c[i]);
//				Mnist.crossValidation(train, param, 5, k[j]);
			}
		}
		
		
		
		param.setC(0.03125);
		
		for(int i = 0; i < k.length; i++) {
//			double accuracy = Mnist.getKnnLabels(train, param, k[i]);
//			System.out.println("k = " + k[i] + ", accuracy = " + accuracy);
		}
		
//		DataPoint[] w = Mnist.train(train, param);
		
		int[] pre = Mnist.knnPredict(train, param, test.x, 81);
		double counter = 0;
		for(int i = 0; i < pre.length; i++) {
			if(pre[i] == test.y[i][0]) {
				counter++;
			}
		}
		
		double accuracy = counter / test.l;
		System.out.println("Accuracy = " + accuracy);
	}

}
