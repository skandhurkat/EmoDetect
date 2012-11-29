//    This file is part of EmoDetect.
//
//    EmoDetect is free software: you can redistribute it and/or modify
//    it under the terms of the GNU General Public License as published by
//    the Free Software Foundation, either version 3 of the License, or
//    (at your option) any later version.
//
//    EmoDetect is distributed in the hope that it will be useful,
//    but WITHOUT ANY WARRANTY; without even the implied warranty of
//    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//    GNU General Public License for more details.
//
//    You should have received a copy of the GNU General Public License
//    along with EmoDetect. If not, see <http://www.gnu.org/licenses/>.
#include <Windows.h>
#include <vector>
#include <iostream>
#include <cstdio>
#include <string>
#define IMAGE_FILE_JPG "jpg"
#define IMAGE_DIR "images"

using namespace std;

void getSubdirs(std::vector<std::string>& output, const std::string& path, bool lookingForDir)
{
    WIN32_FIND_DATA findfiledata;
    HANDLE hFind = INVALID_HANDLE_VALUE;

    char fullpath[MAX_PATH];
    GetFullPathName(path.c_str(), MAX_PATH, fullpath, 0);
    std::string fp(fullpath);

    hFind = FindFirstFile((LPCSTR)(fp + "\\*").c_str(), &findfiledata);
    if (hFind != INVALID_HANDLE_VALUE)
    {
        do
        {
            if(lookingForDir)
            {
                if ((findfiledata.dwFileAttributes | FILE_ATTRIBUTE_DIRECTORY) == FILE_ATTRIBUTE_DIRECTORY
                        && (findfiledata.cFileName[0] != '.'))
                {
                    output.push_back(findfiledata.cFileName);
                }
            }
            else
            {
                if ((findfiledata.dwFileAttributes | FILE_ATTRIBUTE_DIRECTORY) != FILE_ATTRIBUTE_DIRECTORY)
                {
                    int size = string(findfiledata.cFileName).size();
                    string t = string(findfiledata.cFileName).substr(size-3);
                    if(t == IMAGE_FILE_JPG)
                    {
                        output.push_back(findfiledata.cFileName);
                    }
                }
            }
        }
        while (FindNextFile(hFind, &findfiledata) != 0);
    }
}

void readFiles(string dir)
{
    vector <string> subdirs;

    /*
        subdirs - contains all the subdirectores under the IMAGE_DIR directory
    */

    getSubdirs(subdirs, dir, true);

    for(int i=0; i<subdirs.size(); i++)
    {
        vector <string> imageFiles;

        /*
            For each subdirectory imageFiles stores the names of all jpg images
        */

        string subdirname = string(IMAGE_DIR) + "/" + subdirs[i];
        getSubdirs(imageFiles, subdirname, false);

        /*
            Printing it out for convenience
            Business Logic should ideally come in this part of the code
        */

        for(int j=0; j<imageFiles.size(); j++)
        {
            cout << i << " " << (string(IMAGE_DIR) + "/" + subdirs[i] + "/" + imageFiles[j]) << endl;
        }
    }
}

int main()
{
    /*
        IMAGE_DIR is a macro specifying the root image directory path
    */

    freopen("outputLog.txt", "w", stdout);
    readFiles(IMAGE_DIR);
    return 0;
}
