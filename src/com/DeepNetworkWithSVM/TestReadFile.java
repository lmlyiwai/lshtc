package com.DeepNetworkWithSVM;

import java.io.File;
import java.io.IOException;
import java.util.Map;

import org.junit.Test;

import com.IMCLEF.ProcessIMCLEF;
import com.fileInputOutput.InvalidInputDataException;
import com.fileInputOutput.Parameter;
import com.fileInputOutput.Problem;
import com.structure.Structure;

public class TestReadFile {

	@Test
	public void test() throws IOException, InvalidInputDataException {
		String trainfile = "F:\\DataSets\\test.txt";

		Parameter param = new Parameter(10, 1000, 0.001);
		Parameter param1 = new Parameter(1000, 1000, 0.001);
		
		Problem train = Problem.readProblem(new File(trainfile), 1);

		
		Parameter[] params = {param, param1};
		TrainDeepSVM tds = new TrainDeepSVM(train, params, 2);
		String outputFileBase = "F:\\DataSets\\test";
		tds.train(outputFileBase);

		int[][] pre = tds.predict(train.x, outputFileBase);
		for (int i = 0; i < pre.length; i++) {
			for (int j = 0; j < pre[i].length; j++) {
				System.out.print(pre[i][j] + " ");
			}
			System.out.println();
		}
	}

}
