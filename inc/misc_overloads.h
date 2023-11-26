#ifndef MISC_OVERLOADS_H
#define MISC_OVERLOADS_H

#include <list>

template <typename T>
bool operator==(std::list<T> a, std::list<T> b) {
	if(a.size()!=b.size()) {return false;}
	if(a.size() == 0 && b.size() == 0) {return true;}

	for(auto i=a.cbegin(),j=b.cbegin(); i!=a.cend() && j!=b.cend(); ++i,++j) {
		if((*i)!=(*j)) {return false;}
	}
	return true;
}

#endif // MISC_OVERLOADS_H