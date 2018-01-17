package com.simulate;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.OutputStreamWriter;

import org.junit.Test;

import com.fileInputOutput.InvalidInputDataException;
import com.fileInputOutput.Problem;

public class TestScene {

	@Test
	public void test() throws IOException, InvalidInputDataException {
		String trianfile = "F:\\DataSets\\scene\\scene-train.svm";
		String testfile = "F:\\DataSets\\scene\\scene-test.svm";
		
		Problem train = Problem.readProblem(new File(trianfile), 1);
		
		EuclideanDistance ed = new EuclideanDistance(train);
		ed.train(train, 0.001, 1000, 0.001);
		
		double[][] pv = ed.predictValues(train.x);
		ed.sigmoid(pv);
		
		String outfile = "train_predict.txt";
		String testofile = "test_predict.txt";
		BufferedWriter out = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(outfile)));
		String line = null;
		for(int i = 0; i < pv.length; i++) {
			line = new String();
			for(int j = 0; j < pv[i].length; j++) {
				line += pv[i][j] + " ";
			}
			line += "\n";
			out.write(line);
		}
		out.close();
	}

}
