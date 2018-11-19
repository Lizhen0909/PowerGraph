/*  
 * Copyright (c) 2009 Carnegie Mellon University. 
 *     All rights reserved.
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing,
 *  software distributed under the License is distributed on an "AS
 *  IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
 *  express or implied.  See the License for the specific language
 *  governing permissions and limitations under the License.
 *
 *
 */

#include <graphlab.hpp>

bool graph_is_directed = false;
bool graph_is_weighted = false;

// The vertex data is its label 
typedef int vertex_data_type;
typedef float edge_data_type;


// The graph type is determined by the vertex and edge data types
typedef graphlab::distributed_graph<vertex_data_type, edge_data_type> graph_type;

bool line_parser(graph_type& graph, const std::string& filename,
		const std::string& textline) {
	std::stringstream strm(textline);
	graphlab::vertex_id_type vid1;
	graphlab::vertex_id_type vid2;
	edge_data_type weight = 1;
	// first entry in the line is a vertex ID
	strm >> vid1;
	strm >> vid2;
	if (graph_is_weighted)
		strm >> weight;
	// insert this vertex with its label
	graph.add_vertex(vid1, vid1);
	graph.add_vertex(vid2, vid2);
	graph.add_edge(vid1, vid2, weight);

	return true;
}


int main(int argc, char** argv) {
	// Initialize control plain using mpi
	graphlab::mpi_tools::init(argc, argv);
	graphlab::distributed_control dc;
	global_logger().set_log_level(LOG_INFO);

	// Parse command line options -----------------------------------------------
	graphlab::command_line_options clopts("graph to binary");
	std::string graph_dir;
	std::string execution_type = "synchronous";
	clopts.attach_option("graph", graph_dir, "The graph file. Required ");
	clopts.add_positional("graph");
	clopts.attach_option("execution", execution_type,
			"Execution type (synchronous or asynchronous)");
	clopts.attach_option("directed", graph_is_directed, "directed graph.");
	clopts.attach_option("weighted", graph_is_weighted, "weighted graph.");
	std::string saveprefix;
	clopts.attach_option("saveprefix", saveprefix,
			"If set, will save the resultant pagerank to a "
					"sequence of files with prefix saveprefix");

	if (!clopts.parse(argc, argv)) {
		dc.cout() << "Error in parsing command line arguments." << std::endl;
		return EXIT_FAILURE;
	}
	if (graph_dir == "") {
		dc.cout() << "Graph not specified. Cannot continue";
		return EXIT_FAILURE;
	}

	if (saveprefix == "") {
		dc.cout() << "saveprefix not specified. Cannot continue";
		return EXIT_FAILURE;
	}

	// Build the graph ----------------------------------------------------------
	graph_type graph(dc);
	//dc.cout() << "Loading graph in format: " << "snap" << std::endl;
	//graph.load_format(graph_dir, "snap");
	dc.cout() << "Loading graph using line parser" << std::endl;
	graph.load(graph_dir, line_parser);
	// must call finalize before querying the graph
	graph.finalize();

	dc.cout() << "#vertices: " << graph.num_vertices() << " #edges:"
			<< graph.num_edges() << " weighted:" << graph_is_weighted
			<< " directed:" << graph_is_directed << std::endl;


	graph.save_binary(saveprefix);

	graphlab::mpi_tools::finalize();
	return EXIT_SUCCESS;
}
