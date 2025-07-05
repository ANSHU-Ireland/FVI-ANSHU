'use client';

import { useState, useRef, useEffect } from 'react';
import { Card, CardContent } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { apiClient, type ChatMessage } from '@/lib/api';
import { formatDate } from '@/lib/utils';
import { Send, User, Bot, Loader2, ExternalLink } from 'lucide-react';
import ReactMarkdown from 'react-markdown';

export function ChatInterface() {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [currentMessage, setCurrentMessage] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [isStreaming, setIsStreaming] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleSendMessage = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!currentMessage.trim() || isLoading) return;

    const userMessage: ChatMessage = {
      role: 'user',
      content: currentMessage,
      timestamp: new Date().toISOString(),
    };

    setMessages(prev => [...prev, userMessage]);
    setCurrentMessage('');
    setIsLoading(true);

    try {
      // Start streaming response
      setIsStreaming(true);
      const assistantMessage: ChatMessage = {
        role: 'assistant',
        content: '',
        timestamp: new Date().toISOString(),
      };
      setMessages(prev => [...prev, assistantMessage]);

      // Stream the response
      const messageHistory = [...messages, userMessage];
      const stream = apiClient.streamChatMessage(userMessage.content, messageHistory);

      for await (const chunk of stream) {
        setMessages(prev => {
          const newMessages = [...prev];
          const lastMessage = newMessages[newMessages.length - 1];
          if (lastMessage.role === 'assistant') {
            lastMessage.content += chunk;
          }
          return newMessages;
        });
      }

      setIsStreaming(false);
    } catch (error) {
      console.error('Chat error:', error);
      setMessages(prev => {
        const newMessages = [...prev];
        const lastMessage = newMessages[newMessages.length - 1];
        if (lastMessage.role === 'assistant') {
          lastMessage.content = 'I apologize, but I encountered an error while processing your request. Please try again.';
        }
        return newMessages;
      });
      setIsStreaming(false);
    } finally {
      setIsLoading(false);
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage(e);
    }
  };

  return (
    <div className="flex flex-col h-full max-h-[500px]">
      {/* Chat Messages */}
      <div className="flex-1 overflow-y-auto p-4 space-y-4">
        {messages.length === 0 && (
          <div className="text-center text-muted-foreground py-8">
            <Bot className="h-12 w-12 mx-auto mb-4 opacity-50" />
            <h3 className="text-lg font-medium mb-2">Welcome to FVI Assistant</h3>
            <p className="text-sm">
              Ask me anything about your mining operations, metrics, or data insights.
            </p>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-2 mt-4 max-w-md mx-auto">
              <button
                onClick={() => setCurrentMessage("Show me the latest production metrics")}
                className="text-xs p-2 border rounded hover:bg-muted transition-colors"
              >
                Latest production metrics
              </button>
              <button
                onClick={() => setCurrentMessage("What are the top performing mines?")}
                className="text-xs p-2 border rounded hover:bg-muted transition-colors"
              >
                Top performing mines
              </button>
              <button
                onClick={() => setCurrentMessage("Explain the data quality scores")}
                className="text-xs p-2 border rounded hover:bg-muted transition-colors"
              >
                Data quality explanation
              </button>
              <button
                onClick={() => setCurrentMessage("How do dynamic weights work?")}
                className="text-xs p-2 border rounded hover:bg-muted transition-colors"
              >
                Dynamic weights info
              </button>
            </div>
          </div>
        )}

        {messages.map((message, index) => (
          <div
            key={index}
            className={`flex gap-3 ${
              message.role === 'user' ? 'justify-end' : 'justify-start'
            }`}
          >
            {message.role === 'assistant' && (
              <div className="flex-shrink-0 w-8 h-8 rounded-full bg-blue-100 flex items-center justify-center">
                <Bot className="h-4 w-4 text-blue-600" />
              </div>
            )}
            
            <div className={`max-w-[80%] ${
              message.role === 'user' ? 'order-2' : 'order-1'
            }`}>
              <Card className={`${
                message.role === 'user' 
                  ? 'bg-blue-50 border-blue-200' 
                  : 'bg-gray-50 border-gray-200'
              }`}>
                <CardContent className="p-3">
                  <div className="prose prose-sm max-w-none">
                    {message.role === 'user' ? (
                      <p className="text-sm">{message.content}</p>
                    ) : (
                      <ReactMarkdown 
                        className="text-sm"
                        components={{
                          a: ({ node, ...props }) => (
                            <a 
                              {...props} 
                              className="text-blue-600 hover:text-blue-800 inline-flex items-center gap-1"
                              target="_blank"
                              rel="noopener noreferrer"
                            >
                              {props.children}
                              <ExternalLink className="h-3 w-3" />
                            </a>
                          ),
                        }}
                      >
                        {message.content || (isStreaming ? 'Thinking...' : '')}
                      </ReactMarkdown>
                    )}
                  </div>
                  <div className="flex items-center gap-2 mt-2">
                    <Badge variant="outline" className="text-xs">
                      {formatDate(message.timestamp)}
                    </Badge>
                    {message.trace_id && (
                      <Badge variant="outline" className="text-xs font-mono">
                        {message.trace_id.slice(0, 8)}
                      </Badge>
                    )}
                  </div>
                </CardContent>
              </Card>
            </div>

            {message.role === 'user' && (
              <div className="flex-shrink-0 w-8 h-8 rounded-full bg-green-100 flex items-center justify-center order-3">
                <User className="h-4 w-4 text-green-600" />
              </div>
            )}
          </div>
        ))}

        {isStreaming && (
          <div className="flex justify-start">
            <div className="flex-shrink-0 w-8 h-8 rounded-full bg-blue-100 flex items-center justify-center">
              <Loader2 className="h-4 w-4 text-blue-600 animate-spin" />
            </div>
          </div>
        )}

        <div ref={messagesEndRef} />
      </div>

      {/* Input Form */}
      <div className="border-t p-4">
        <form onSubmit={handleSendMessage} className="flex gap-2">
          <textarea
            ref={textareaRef}
            value={currentMessage}
            onChange={(e) => setCurrentMessage(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Ask me anything about your mining operations..."
            className="flex-1 min-h-[44px] max-h-[120px] p-2 border rounded-md resize-none focus:outline-none focus:ring-2 focus:ring-blue-500"
            disabled={isLoading}
          />
          <button
            type="submit"
            disabled={isLoading || !currentMessage.trim()}
            className="px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-2"
          >
            {isLoading ? (
              <Loader2 className="h-4 w-4 animate-spin" />
            ) : (
              <Send className="h-4 w-4" />
            )}
            Send
          </button>
        </form>
      </div>
    </div>
  );
}
