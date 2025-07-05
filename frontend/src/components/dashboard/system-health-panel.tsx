'use client';

import { useQuery } from '@tanstack/react-query';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { apiClient } from '@/lib/api';
import { formatPercentage, formatNumber } from '@/lib/utils';
import { Activity, Cpu, HardDrive, Zap, Users, AlertTriangle } from 'lucide-react';
import { ResponsiveContainer, LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, AreaChart, Area } from 'recharts';

export function SystemHealthPanel() {
  const { data: systemMetrics, isLoading } = useQuery({
    queryKey: ['system-metrics'],
    queryFn: () => apiClient.getSystemMetrics(),
    refetchInterval: 5000, // Refetch every 5 seconds
  });

  const { data: healthData } = useQuery({
    queryKey: ['health'],
    queryFn: () => apiClient.healthCheck(),
    refetchInterval: 10000,
  });

  if (isLoading) {
    return (
      <Card>
        <CardHeader>
          <CardTitle>System Health</CardTitle>
          <CardDescription>Loading system metrics...</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="h-[400px] flex items-center justify-center">
            <div className="text-sm text-muted-foreground">Loading...</div>
          </div>
        </CardContent>
      </Card>
    );
  }

  if (!systemMetrics?.data) {
    return (
      <Card>
        <CardHeader>
          <CardTitle>System Health</CardTitle>
          <CardDescription>No system metrics available</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="h-[400px] flex items-center justify-center">
            <div className="text-sm text-muted-foreground">No data available</div>
          </div>
        </CardContent>
      </Card>
    );
  }

  const metrics = systemMetrics.data;

  const getHealthColor = (value: number, threshold: number) => {
    if (value >= threshold) return 'text-red-600';
    if (value >= threshold * 0.8) return 'text-yellow-600';
    return 'text-green-600';
  };

  const getHealthBadge = (value: number, threshold: number) => {
    if (value >= threshold) return 'bg-red-100 text-red-800';
    if (value >= threshold * 0.8) return 'bg-yellow-100 text-yellow-800';
    return 'bg-green-100 text-green-800';
  };

  // Mock historical data for demonstration
  const historicalData = Array.from({ length: 24 }, (_, i) => ({
    hour: i,
    cpu: Math.max(0, metrics.cpu_usage + (Math.random() - 0.5) * 20),
    memory: Math.max(0, metrics.memory_usage + (Math.random() - 0.5) * 15),
    disk: Math.max(0, metrics.disk_usage + (Math.random() - 0.5) * 10),
  }));

  return (
    <div className="space-y-6">
      {/* System Status Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">CPU Usage</CardTitle>
            <Cpu className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className={`text-2xl font-bold ${getHealthColor(metrics.cpu_usage, 80)}`}>
              {formatPercentage(metrics.cpu_usage / 100)}
            </div>
            <Badge className={`mt-2 ${getHealthBadge(metrics.cpu_usage, 80)}`}>
              {metrics.cpu_usage >= 80 ? 'High' : metrics.cpu_usage >= 60 ? 'Moderate' : 'Low'}
            </Badge>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Memory Usage</CardTitle>
            <Zap className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className={`text-2xl font-bold ${getHealthColor(metrics.memory_usage, 85)}`}>
              {formatPercentage(metrics.memory_usage / 100)}
            </div>
            <Badge className={`mt-2 ${getHealthBadge(metrics.memory_usage, 85)}`}>
              {metrics.memory_usage >= 85 ? 'High' : metrics.memory_usage >= 70 ? 'Moderate' : 'Low'}
            </Badge>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Disk Usage</CardTitle>
            <HardDrive className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className={`text-2xl font-bold ${getHealthColor(metrics.disk_usage, 90)}`}>
              {formatPercentage(metrics.disk_usage / 100)}
            </div>
            <Badge className={`mt-2 ${getHealthBadge(metrics.disk_usage, 90)}`}>
              {metrics.disk_usage >= 90 ? 'High' : metrics.disk_usage >= 75 ? 'Moderate' : 'Low'}
            </Badge>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Active Connections</CardTitle>
            <Users className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">
              {formatNumber(metrics.active_connections)}
            </div>
            <p className="text-xs text-muted-foreground">
              Current connections
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Request Rate</CardTitle>
            <Activity className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">
              {formatNumber(metrics.request_rate)}
            </div>
            <p className="text-xs text-muted-foreground">
              Requests per second
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Error Rate</CardTitle>
            <AlertTriangle className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className={`text-2xl font-bold ${getHealthColor(metrics.error_rate, 5)}`}>
              {formatPercentage(metrics.error_rate / 100)}
            </div>
            <Badge className={`mt-2 ${getHealthBadge(metrics.error_rate, 5)}`}>
              {metrics.error_rate >= 5 ? 'High' : metrics.error_rate >= 2 ? 'Moderate' : 'Low'}
            </Badge>
          </CardContent>
        </Card>
      </div>

      {/* Historical Trends */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Activity className="h-5 w-5" />
            Resource Usage Trends
          </CardTitle>
          <CardDescription>
            24-hour historical view of system resource utilization
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="h-[300px]">
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart data={historicalData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="hour" />
                <YAxis domain={[0, 100]} />
                <Tooltip 
                  formatter={(value) => [`${value.toFixed(1)}%`, '']}
                />
                <Area 
                  type="monotone" 
                  dataKey="cpu" 
                  stackId="1" 
                  stroke="#8884d8" 
                  fill="#8884d8" 
                  fillOpacity={0.6}
                />
                <Area 
                  type="monotone" 
                  dataKey="memory" 
                  stackId="1" 
                  stroke="#82ca9d" 
                  fill="#82ca9d" 
                  fillOpacity={0.6}
                />
                <Area 
                  type="monotone" 
                  dataKey="disk" 
                  stackId="1" 
                  stroke="#ffc658" 
                  fill="#ffc658" 
                  fillOpacity={0.6}
                />
              </AreaChart>
            </ResponsiveContainer>
          </div>
        </CardContent>
      </Card>

      {/* System Status Summary */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Activity className="h-5 w-5" />
            System Status Summary
          </CardTitle>
          <CardDescription>
            Overall system health and recommendations
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            {/* Overall Status */}
            <div className="flex items-center justify-between p-4 bg-muted rounded-lg">
              <div className="flex items-center gap-3">
                <div className={`w-3 h-3 rounded-full ${
                  healthData?.data?.status === 'healthy' ? 'bg-green-500' : 'bg-red-500'
                }`} />
                <div>
                  <div className="font-medium">System Status</div>
                  <div className="text-sm text-muted-foreground">
                    {healthData?.data?.status || 'Unknown'}
                  </div>
                </div>
              </div>
              <Badge variant={healthData?.data?.status === 'healthy' ? 'default' : 'destructive'}>
                {healthData?.data?.status || 'Unknown'}
              </Badge>
            </div>

            {/* Recommendations */}
            <div className="space-y-2">
              <h4 className="text-sm font-medium">Recommendations</h4>
              <div className="space-y-2">
                {metrics.cpu_usage > 80 && (
                  <div className="flex items-center gap-2 text-sm text-yellow-600">
                    <AlertTriangle className="h-4 w-4" />
                    <span>High CPU usage detected - consider scaling resources</span>
                  </div>
                )}
                {metrics.memory_usage > 85 && (
                  <div className="flex items-center gap-2 text-sm text-yellow-600">
                    <AlertTriangle className="h-4 w-4" />
                    <span>High memory usage - monitor for memory leaks</span>
                  </div>
                )}
                {metrics.disk_usage > 90 && (
                  <div className="flex items-center gap-2 text-sm text-red-600">
                    <AlertTriangle className="h-4 w-4" />
                    <span>Critical disk usage - cleanup or expand storage</span>
                  </div>
                )}
                {metrics.error_rate > 5 && (
                  <div className="flex items-center gap-2 text-sm text-red-600">
                    <AlertTriangle className="h-4 w-4" />
                    <span>High error rate - investigate application issues</span>
                  </div>
                )}
                {metrics.cpu_usage <= 80 && metrics.memory_usage <= 85 && metrics.disk_usage <= 90 && metrics.error_rate <= 5 && (
                  <div className="flex items-center gap-2 text-sm text-green-600">
                    <Activity className="h-4 w-4" />
                    <span>All systems operating within normal parameters</span>
                  </div>
                )}
              </div>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
