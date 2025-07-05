'use client';

import { useQuery } from '@tanstack/react-query';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { apiClient } from '@/lib/api';
import { formatPercentage } from '@/lib/utils';
import { Shield, CheckCircle, AlertTriangle, XCircle, TrendingUp } from 'lucide-react';
import { ResponsiveContainer, PieChart, Pie, Cell, Tooltip, BarChart, Bar, XAxis, YAxis, CartesianGrid } from 'recharts';

export function DataQualityPanel() {
  const { data: dataQuality, isLoading } = useQuery({
    queryKey: ['data-quality'],
    queryFn: () => apiClient.getDataQualityMetrics(),
    refetchInterval: 30000, // Refetch every 30 seconds
  });

  if (isLoading) {
    return (
      <Card>
        <CardHeader>
          <CardTitle>Data Quality Overview</CardTitle>
          <CardDescription>Loading data quality metrics...</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="h-[300px] flex items-center justify-center">
            <div className="text-sm text-muted-foreground">Loading...</div>
          </div>
        </CardContent>
      </Card>
    );
  }

  if (!dataQuality?.data) {
    return (
      <Card>
        <CardHeader>
          <CardTitle>Data Quality Overview</CardTitle>
          <CardDescription>No data quality metrics available</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="h-[300px] flex items-center justify-center">
            <div className="text-sm text-muted-foreground">No data available</div>
          </div>
        </CardContent>
      </Card>
    );
  }

  const qualityMetrics = [
    {
      name: 'Completeness',
      value: dataQuality.data.completeness,
      color: '#8884d8',
      icon: CheckCircle,
    },
    {
      name: 'Accuracy',
      value: dataQuality.data.accuracy,
      color: '#82ca9d',
      icon: Shield,
    },
    {
      name: 'Consistency',
      value: dataQuality.data.consistency,
      color: '#ffc658',
      icon: TrendingUp,
    },
    {
      name: 'Timeliness',
      value: dataQuality.data.timeliness,
      color: '#ff7c7c',
      icon: AlertTriangle,
    },
    {
      name: 'Validity',
      value: dataQuality.data.validity,
      color: '#8dd1e1',
      icon: XCircle,
    },
  ];

  const chartData = qualityMetrics.map(metric => ({
    name: metric.name,
    value: metric.value * 100,
    color: metric.color,
  }));

  const overallScore = qualityMetrics.reduce((acc, metric) => acc + metric.value, 0) / qualityMetrics.length;

  const getQualityColor = (score: number) => {
    if (score >= 0.9) return 'text-green-600';
    if (score >= 0.7) return 'text-yellow-600';
    return 'text-red-600';
  };

  const getQualityBadge = (score: number) => {
    if (score >= 0.9) return 'bg-green-100 text-green-800';
    if (score >= 0.7) return 'bg-yellow-100 text-yellow-800';
    return 'bg-red-100 text-red-800';
  };

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Shield className="h-5 w-5" />
          Data Quality Overview
        </CardTitle>
        <CardDescription>
          Real-time data quality metrics across all data sources
        </CardDescription>
      </CardHeader>
      <CardContent>
        <div className="space-y-6">
          {/* Overall Score */}
          <div className="text-center p-4 bg-muted rounded-lg">
            <div className={`text-3xl font-bold ${getQualityColor(overallScore)}`}>
              {formatPercentage(overallScore)}
            </div>
            <div className="text-sm text-muted-foreground mt-1">Overall Quality Score</div>
            <Badge className={`mt-2 ${getQualityBadge(overallScore)}`}>
              {overallScore >= 0.9 ? 'Excellent' : overallScore >= 0.7 ? 'Good' : 'Needs Improvement'}
            </Badge>
          </div>

          {/* Quality Metrics Chart */}
          <div className="h-[200px]">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={chartData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="name" />
                <YAxis domain={[0, 100]} />
                <Tooltip 
                  formatter={(value) => [`${value}%`, 'Score']}
                />
                <Bar dataKey="value" fill="#8884d8" />
              </BarChart>
            </ResponsiveContainer>
          </div>

          {/* Detailed Metrics */}
          <div className="grid grid-cols-1 gap-3">
            {qualityMetrics.map((metric) => {
              const Icon = metric.icon;
              return (
                <div key={metric.name} className="flex items-center justify-between p-3 border rounded-lg">
                  <div className="flex items-center gap-3">
                    <Icon className="h-5 w-5 text-muted-foreground" />
                    <div>
                      <div className="font-medium">{metric.name}</div>
                      <div className="text-sm text-muted-foreground">
                        {metric.name === 'Completeness' && 'Percentage of complete data records'}
                        {metric.name === 'Accuracy' && 'Correctness of data values'}
                        {metric.name === 'Consistency' && 'Uniformity across data sources'}
                        {metric.name === 'Timeliness' && 'Freshness of data updates'}
                        {metric.name === 'Validity' && 'Conformance to data standards'}
                      </div>
                    </div>
                  </div>
                  <div className="text-right">
                    <div className={`text-lg font-bold ${getQualityColor(metric.value)}`}>
                      {formatPercentage(metric.value)}
                    </div>
                    <div className="text-xs text-muted-foreground">
                      {metric.value >= 0.9 ? 'Excellent' : metric.value >= 0.7 ? 'Good' : 'Poor'}
                    </div>
                  </div>
                </div>
              );
            })}
          </div>

          {/* Quality Trends */}
          <div className="pt-4 border-t">
            <h4 className="text-sm font-medium mb-3">Quality Insights</h4>
            <div className="space-y-2">
              {overallScore >= 0.9 && (
                <div className="flex items-center gap-2 text-sm text-green-600">
                  <CheckCircle className="h-4 w-4" />
                  <span>All data quality metrics are performing excellently</span>
                </div>
              )}
              {overallScore < 0.9 && dataQuality.data.completeness < 0.8 && (
                <div className="flex items-center gap-2 text-sm text-yellow-600">
                  <AlertTriangle className="h-4 w-4" />
                  <span>Data completeness needs attention - consider additional data sources</span>
                </div>
              )}
              {overallScore < 0.7 && (
                <div className="flex items-center gap-2 text-sm text-red-600">
                  <XCircle className="h-4 w-4" />
                  <span>Multiple quality issues detected - review data pipeline</span>
                </div>
              )}
            </div>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}
