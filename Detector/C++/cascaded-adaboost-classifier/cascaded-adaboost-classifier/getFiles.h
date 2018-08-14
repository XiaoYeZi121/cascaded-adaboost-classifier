#include <glob.h>
#include <vector>
#include <string>
std::vector<std::string> getFiles(const std::string& pat)
{
   using namespace std;
   glob_t glob_result;
   glob(pat.c_str(), GLOB_TILDE, NULL, &glob_result);
   vector<string> ret;
   for(unsigned int i=0; i<glob_result.gl_pathc; ++i)
   {
       ret.push_back(string(glob_result.gl_pathv[i]));
   }
   globfree(&glob_result);
   return ret;
}
