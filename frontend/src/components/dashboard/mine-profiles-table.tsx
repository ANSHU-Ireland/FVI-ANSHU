'use client';

import { useQuery } from '@tanstack/react-query';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { apiClient, type MineProfile } from '@/lib/api';
import { formatDate, getStatusBadgeColor } from '@/lib/utils';
import { MapPin, Factory, Calendar, Activity } from 'lucide-react';

export function MineProfilesTable() {
  const { data: mines, isLoading, error } = useQuery({
    queryKey: ['mine-profiles'],
    queryFn: () => apiClient.getMineProfiles({ limit: 50 }),
    refetchInterval: 60000, // Refetch every minute
  });

  if (isLoading) {
    return (
      <Card>
        <CardHeader>
          <CardTitle>Mine Profiles</CardTitle>
          <CardDescription>Loading mine data...</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="h-[400px] flex items-center justify-center">
            <div className="text-sm text-muted-foreground">Loading...</div>
          </div>
        </CardContent>
      </Card>
    );
  }

  if (error) {
    return (
      <Card>
        <CardHeader>
          <CardTitle>Mine Profiles</CardTitle>
          <CardDescription>Error loading mine data</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="h-[400px] flex items-center justify-center">
            <div className="text-sm text-red-600">
              Failed to load mine profiles: {error.message}
            </div>
          </div>
        </CardContent>
      </Card>
    );
  }

  if (!mines?.data?.length) {
    return (
      <Card>
        <CardHeader>
          <CardTitle>Mine Profiles</CardTitle>
          <CardDescription>No mine profiles found</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="h-[400px] flex items-center justify-center">
            <div className="text-sm text-muted-foreground">No data available</div>
          </div>
        </CardContent>
      </Card>
    );
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Factory className="h-5 w-5" />
          Mine Profiles
        </CardTitle>
        <CardDescription>
          Overview of all mining operations ({mines.data.length} total)
        </CardDescription>
      </CardHeader>
      <CardContent>
        <div className="space-y-4">
          {/* Summary Stats */}
          <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
            <div className="text-center p-3 bg-muted rounded-lg">
              <div className="text-2xl font-bold text-blue-600">
                {mines.data.length}
              </div>
              <div className="text-sm text-muted-foreground">Total Mines</div>
            </div>
            <div className="text-center p-3 bg-muted rounded-lg">
              <div className="text-2xl font-bold text-green-600">
                {mines.data.filter(m => m.status.toLowerCase() === 'active').length}
              </div>
              <div className="text-sm text-muted-foreground">Active</div>
            </div>
            <div className="text-center p-3 bg-muted rounded-lg">
              <div className="text-2xl font-bold text-orange-600">
                {new Set(mines.data.map(m => m.commodity)).size}
              </div>
              <div className="text-sm text-muted-foreground">Commodities</div>
            </div>
            <div className="text-center p-3 bg-muted rounded-lg">
              <div className="text-2xl font-bold text-purple-600">
                {mines.data.reduce((acc, m) => acc + m.production_capacity, 0).toLocaleString()}
              </div>
              <div className="text-sm text-muted-foreground">Total Capacity</div>
            </div>
          </div>

          {/* Mine Profiles Table */}
          <div className="overflow-x-auto">
            <table className="w-full border-collapse">
              <thead>
                <tr className="border-b">
                  <th className="text-left p-3 font-medium">Mine</th>
                  <th className="text-left p-3 font-medium">Location</th>
                  <th className="text-left p-3 font-medium">Commodity</th>
                  <th className="text-left p-3 font-medium">Capacity</th>
                  <th className="text-left p-3 font-medium">Status</th>
                  <th className="text-left p-3 font-medium">Last Updated</th>
                </tr>
              </thead>
              <tbody>
                {mines.data.map((mine: MineProfile) => (
                  <tr key={mine.mine_id} className="border-b hover:bg-muted/50">
                    <td className="p-3">
                      <div className="flex items-center gap-2">
                        <Factory className="h-4 w-4 text-muted-foreground" />
                        <div>
                          <div className="font-medium">{mine.name}</div>
                          <div className="text-sm text-muted-foreground">
                            {mine.mine_id}
                          </div>
                        </div>
                      </div>
                    </td>
                    <td className="p-3">
                      <div className="flex items-center gap-2">
                        <MapPin className="h-4 w-4 text-muted-foreground" />
                        <span className="text-sm">{mine.location}</span>
                      </div>
                    </td>
                    <td className="p-3">
                      <Badge variant="secondary" className="text-xs">
                        {mine.commodity}
                      </Badge>
                    </td>
                    <td className="p-3">
                      <div className="text-sm font-medium">
                        {mine.production_capacity.toLocaleString()}
                      </div>
                      <div className="text-xs text-muted-foreground">units/year</div>
                    </td>
                    <td className="p-3">
                      <Badge className={getStatusBadgeColor(mine.status)}>
                        <Activity className="h-3 w-3 mr-1" />
                        {mine.status}
                      </Badge>
                    </td>
                    <td className="p-3">
                      <div className="flex items-center gap-2">
                        <Calendar className="h-4 w-4 text-muted-foreground" />
                        <span className="text-sm">
                          {formatDate(mine.last_updated)}
                        </span>
                      </div>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          {/* Commodity Breakdown */}
          <div className="mt-6">
            <h4 className="text-sm font-medium mb-3">Commodity Breakdown</h4>
            <div className="grid grid-cols-1 md:grid-cols-3 lg:grid-cols-5 gap-3">
              {Object.entries(
                mines.data.reduce((acc, mine) => {
                  acc[mine.commodity] = (acc[mine.commodity] || 0) + 1;
                  return acc;
                }, {} as Record<string, number>)
              ).map(([commodity, count]) => (
                <div key={commodity} className="text-center p-3 bg-muted rounded-lg">
                  <div className="text-lg font-bold">{count}</div>
                  <div className="text-sm text-muted-foreground">{commodity}</div>
                </div>
              ))}
            </div>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}
