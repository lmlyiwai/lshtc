package com.recent.test;

import java.io.File;
import java.io.IOException;

import org.junit.Test;

import com.fileInputOutput.InvalidInputDataException;
import com.fileInputOutput.Parameter;
import com.fileInputOutput.Problem;
import com.rssvm.Linear;
import com.sparseVector.DataPoint;

public class TestLinearSVM {

	@Test
	public void test() throws Exception, InvalidInputDataException {
        String filename = "F:\\C#\\test.txt";
        Problem prob = Problem.readProblem(new File(filename), 1);
        Parameter param = new Parameter(10, 1000, 0.001);
        int[] y = { 1, 1, -1, -1 };
        double[] tloss = new double[1];
        DataPoint[] w = Linear.train(prob, y, param, null, tloss, null, 0);
        for(DataPoint dp : w) {
        	System.out.println(dp.index + ": " + dp.value);
        }
	}

}
