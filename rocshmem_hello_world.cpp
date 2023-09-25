#include "roc_shmem.hpp"
#include <iostream>

int main() {
	Team team();
	std::cout << team.get_pe_in_the_world() << std::endl;
	return 0;
}
