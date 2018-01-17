package com.confirm;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.OutputStreamWriter;
import java.util.ArrayList;
import java.util.List;

import com.fileInputOutput.InvalidInputDataException;
import com.fileInputOutput.Problem;
import com.sparseVector.DataPoint;

public class Test {

	@org.junit.Test
	public void test() throws IOException, InvalidInputDataException {
		String trainfile = "F:\\java\\RecursiveRegularizationSVM_1\\scene_train.txt";
		String testfile = "F:\\java\\RecursiveRegularizationSVM_1\\scene_test.txt";
		
		Problem train = Problem.readProblem(new File(trainfile), -1);
		Problem test = Problem.readProblem(new File(testfile), -1);
		
		scale(train.x);
		scale(test.x);
		
		List<DataPoint[]> trainList = new ArrayList<DataPoint[]>();
		List<DataPoint[]> testList = new ArrayList<DataPoint[]>();
		for(int i = 0; i < train.l; i++) {
			if(train.y[i].length == 2 && train.y[i][0] == 0 && train.y[i][1] == 4) {
				trainList.add(train.x[i]);
			}
		}
		
		for(int i = 0; i < test.l; i++) {
			if(test.y[i].length == 2 && test.y[i][0] == 0 && test.y[i][1] == 4) {
				testList.add(test.x[i]);
			}
		}
		
		String train04out = "train04.txt";
		String test04out = "test04.txt";
		writeSamplesToFile(trainList, train04out);
		writeSamplesToFile(testList, test04out);
	}

	public static void scale(DataPoint[][] x) {
		for(int i = 0; i < x.length; i++) {
			double sum = 0;
			for(DataPoint dp : x[i]) {
				sum += dp.value;
			}
			
			for(DataPoint dp: x[i]) {
				dp.value = dp.value / sum;
			}
		}
	}
	
	/**
	 * @throws IOException 
	 * 
	 */
	public static void writeSamplesToFile(List<DataPoint[]> list, String filename) throws IOException {
		BufferedWriter out = new BufferedWriter(new OutputStreamWriter(
				new FileOutputStream(filename)));
		for(int i = 0; i < list.size(); i++) {
			DataPoint[] dp = list.get(i);
			String line = new String();
			for(int j = 0; j < dp.length; j++) {
				line = line + dp[j].value + " ";
			}
			line = line + "\n";
			out.write(line);
		}
		out.close();
	}
}
