package com.test;

import java.io.BufferedWriter;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.OutputStreamWriter;

import org.junit.Test;

public class TestResult {

	@Test
	public void test() throws IOException {
		String filename = "result.txt";
		BufferedWriter out = new BufferedWriter(new OutputStreamWriter(
				new FileOutputStream(filename)));
		for(int i = 0; i < 34880; i++) {
			int t = 21;
			String line = t + "\n";
			out.write(line);
		}
		out.close();
	}

}
