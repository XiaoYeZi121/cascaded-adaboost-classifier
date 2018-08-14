// BoostedCommittee.h: interface for the CBoostedCommittee class.
//
//////////////////////////////////////////////////////////////////////

#if !defined(AFX_BOOSTEDCOMMITTEE_H__BAEE4FC8_5AEF_4B42_854C_845B8CF397B8__INCLUDED_)
#define AFX_BOOSTEDCOMMITTEE_H__BAEE4FC8_5AEF_4B42_854C_845B8CF397B8__INCLUDED_

#if _MSC_VER > 1000
#pragma once
#endif // _MSC_VER > 1000

#include <vector>
#include "SPHypothesis.h"

#include <stdio.h>
class CBoostedCommittee
{
public:
	CBoostedCommittee();
	virtual ~CBoostedCommittee();

  double Predict(double * in_Sample);

  bool LoadFromFile(FILE* in_File);

  bool LoadFromString(const char* Data);

  std::vector <CSPHypothesis> get_m_vHypotheses(){ return m_vHypotheses; }
  std::vector <double> get_m_vWeights(){ return m_vWeights; }

protected:
  std::vector <CSPHypothesis> m_vHypotheses;
  std::vector <double> m_vWeights;
};

#endif // !defined(AFX_BOOSTEDCOMMITTEE_H__BAEE4FC8_5AEF_4B42_854C_845B8CF397B8__INCLUDED_)
