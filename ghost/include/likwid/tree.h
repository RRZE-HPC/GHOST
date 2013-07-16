/*
 * =======================================================================================
 *
 *      Filename:  tree.h
 *
 *      Description:  Header File tree Module. 
 *                    Implements a simple tree data structure.
 *
 *      Version:   3.0
 *      Released:  29.11.2012
 *
 *      Author:  Jan Treibig (jt), jan.treibig@gmail.com
 *      Project:  likwid
 *
 *      Copyright (C) 2012 Jan Treibig 
 *
 *      This program is free software: you can redistribute it and/or modify it under
 *      the terms of the GNU General Public License as published by the Free Software
 *      Foundation, either version 3 of the License, or (at your option) any later
 *      version.
 *
 *      This program is distributed in the hope that it will be useful, but WITHOUT ANY
 *      WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
 *      PARTICULAR PURPOSE.  See the GNU General Public License for more details.
 *
 *      You should have received a copy of the GNU General Public License along with
 *      this program.  If not, see <http://www.gnu.org/licenses/>.
 *
 * =======================================================================================
 */

#ifndef TREE_H
#define TREE_H

#include <types.h>

extern void ghost_tree_init(TreeNode** root, int id);
extern void ghost_tree_print(TreeNode* nodePtr);
extern void ghost_tree_insertNode(TreeNode* nodePtr, int id);
extern int ghost_tree_nodeExists(TreeNode* nodePtr, int id);
extern int ghost_tree_countChildren(TreeNode* nodePtr);
extern void ghost_tree_sort(TreeNode* root);
extern TreeNode* ghost_tree_getNode(TreeNode* nodePtr, int id);
extern TreeNode* ghost_tree_getChildNode(TreeNode* nodePtr);
extern TreeNode* ghost_tree_getNextNode(TreeNode* nodePtr);

#endif /*TREE_H*/
