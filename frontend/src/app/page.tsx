'use client';

import { useQuery } from '@tanstack/react-query';
import { apiClient } from '@/lib/api';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { MetricsOverview } from '@/components/dashboard/metrics-overview';
import { MineProfilesTable } from '@/components/dashboard/mine-profiles-table';
import { WeightMonitor } from '@/components/dashboard/weight-monitor';
import { DataQualityPanel } from '@/components/dashboard/data-quality-panel';
import { SystemHealthPanel } from '@/components/dashboard/system-health-panel';
import { RecentPredictions } from '@/components/dashboard/recent-predictions';
import { ChatInterface } from '@/components/chat/chat-interface';
import { Activity, Database, TrendingUp, Settings, MessageSquare, BarChart3 } from 'lucide-react';

export default function DashboardPage() {
  const { data: healthData, isLoading: healthLoading } = useQuery({
    queryKey: ['health'],
    queryFn: () => apiClient.healthCheck(),
    refetchInterval: 10000, // Refetch every 10 seconds
  });

  const { data: metricsData, isLoading: metricsLoading } = useQuery({
    queryKey: ['metrics-summary'],
    queryFn: () => apiClient.getMetricSummary(),
    refetchInterval: 30000, // Refetch every 30 seconds
  });

  const { data: systemMetrics, isLoading: systemLoading } = useQuery({
    queryKey: ['system-metrics'],
    queryFn: () => apiClient.getSystemMetrics(),
    refetchInterval: 5000, // Refetch every 5 seconds
  });

  return (
    <div className="container mx-auto p-6 space-y-6">
      {/* Header */}
      <div className="flex justify-between items-center">
        <div>
          <h1 className="text-3xl font-bold tracking-tight">FVI Analytics Platform</h1>
          <p className="text-muted-foreground">
            Production-grade analytics for mining operations
          </p>
        </div>
        <div className="flex items-center space-x-2">
          <Badge variant={healthData?.data?.status === 'healthy' ? 'default' : 'destructive'}>
            {healthLoading ? 'Checking...' : healthData?.data?.status || 'Unknown'}
          </Badge>
          {healthData?.data?.timestamp && (
            <span className="text-sm text-muted-foreground">
              Last updated: {new Date(healthData.data.timestamp).toLocaleTimeString()}
            </span>
          )}
        </div>
      </div>

      {/* Quick Stats */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Total Metrics</CardTitle>
            <BarChart3 className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">
              {metricsLoading ? '...' : metricsData?.data?.total_metrics?.toLocaleString() || '0'}
            </div>
            <p className="text-xs text-muted-foreground">
              Across all mining operations
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Active Mines</CardTitle>
            <Activity className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">
              {metricsLoading ? '...' : metricsData?.data?.active_mines || '0'}
            </div>
            <p className="text-xs text-muted-foreground">
              Currently operational
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Avg Trust Score</CardTitle>
            <TrendingUp className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">
              {metricsLoading ? '...' : `${((metricsData?.data?.avg_trust_score || 0) * 100).toFixed(1)}%`}
            </div>
            <p className="text-xs text-muted-foreground">
              Data reliability metric
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Data Quality</CardTitle>
            <Database className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">
              {metricsLoading ? '...' : `${((metricsData?.data?.avg_data_quality_score || 0) * 100).toFixed(1)}%`}
            </div>
            <p className="text-xs text-muted-foreground">
              Overall data quality
            </p>
          </CardContent>
        </Card>
      </div>

      {/* Main Content Tabs */}
      <Tabs defaultValue="overview" className="space-y-4">
        <TabsList className="grid w-full grid-cols-6">
          <TabsTrigger value="overview">Overview</TabsTrigger>
          <TabsTrigger value="mines">Mines</TabsTrigger>
          <TabsTrigger value="predictions">Predictions</TabsTrigger>
          <TabsTrigger value="weights">Weights</TabsTrigger>
          <TabsTrigger value="chat">Chat</TabsTrigger>
          <TabsTrigger value="system">System</TabsTrigger>
        </TabsList>

        <TabsContent value="overview" className="space-y-4">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <MetricsOverview />
            <DataQualityPanel />
          </div>
          <RecentPredictions />
        </TabsContent>

        <TabsContent value="mines" className="space-y-4">
          <MineProfilesTable />
        </TabsContent>

        <TabsContent value="predictions" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>ML Predictions</CardTitle>
              <CardDescription>
                Real-time predictions with explainability
              </CardDescription>
            </CardHeader>
            <CardContent>
              <RecentPredictions showAll />
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="weights" className="space-y-4">
          <WeightMonitor />
        </TabsContent>

        <TabsContent value="chat" className="space-y-4">
          <Card className="h-[600px]">
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <MessageSquare className="h-5 w-5" />
                AI Assistant
              </CardTitle>
              <CardDescription>
                Chat with your data using natural language
              </CardDescription>
            </CardHeader>
            <CardContent className="h-full">
              <ChatInterface />
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="system" className="space-y-4">
          <SystemHealthPanel />
        </TabsContent>
      </Tabs>
    </div>
  );
}
