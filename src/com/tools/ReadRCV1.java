package com.tools;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.HashMap;
import java.util.Map;

public class ReadRCV1 {
	/**
	 * 读入类别标签，并指定编号
	 * @throws IOException 
	 * */
	public static Map<String, Integer> readTopic(String filename) throws IOException {
		BufferedReader in = new BufferedReader(
				new InputStreamReader(new FileInputStream(filename)));
		Map<String, Integer> map = new HashMap<String, Integer>();
		
		String line = null;
		int id = 0;
		while((line = in.readLine()) != null) {
			line = line.trim();
			map.put(line, id++);
		}
		
		in.close();
		return map;
	}
}
