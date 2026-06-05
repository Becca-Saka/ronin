import 'package:flutter/material.dart';

class E2eCounterPageNo extends StatefulWidget {
  const E2eCounterPageNo({super.key});

  @override
  State<E2eCounterPageNo> createState() => _E2eCounterPageNoState();
}

class _E2eCounterPageNoState extends State<E2eCounterPageNo> {
  bool _loggedIn = false;
  int _counter = 0;
  bool _alertsEnabled = true;
  final _emailController = TextEditingController();
  final _searchController = TextEditingController();

  @override
  void dispose() {
    _emailController.dispose();
    _searchController.dispose();
    super.dispose();
  }

  void _increment() {
    setState(() => _counter++);
  }

  void _login() {
    setState(() => _loggedIn = true);
  }

  @override
  Widget build(BuildContext context) {
    if (!_loggedIn) {
      return Scaffold(
        appBar: AppBar(
          title: const Text('Sign in'),
        ),
        body: Padding(
          padding: const EdgeInsets.all(24),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.stretch,
            children: [
              const Text('Welcome back'),
              const SizedBox(height: 24),
              TextField(
                controller: _emailController,
                keyboardType: TextInputType.emailAddress,
                decoration: const InputDecoration(
                  labelText: 'Email',
                  hintText: 'you@example.com',
                  border: OutlineInputBorder(),
                ),
              ),
              const SizedBox(height: 24),
              FilledButton(
                onPressed: _login,
                child: const Text('Login'),
              ),
            ],
          ),
        ),
      );
    }

    return DefaultTabController(
      length: 2,
      child: Scaffold(
        appBar: AppBar(
          title: const Text('Home screen'),
          actions: [
            IconButton(
              icon: const Icon(Icons.search),
              onPressed: () {},
            ),
            IconButton(
              icon: const Icon(Icons.more_vert),
              tooltip: 'More options',
              onPressed: () {},
            ),
          ],
          bottom: const TabBar(
            tabs: [
              Tab(text: 'Home'),
              Tab(text: 'Inbox'),
            ],
          ),
        ),
        body: TabBarView(
          children: [
            _HomeTab(counter: _counter),
            _InboxTab(
              alertsEnabled: _alertsEnabled,
              searchController: _searchController,
              onAlertsChanged: (value) => setState(() => _alertsEnabled = value),
            ),
          ],
        ),
        floatingActionButton: FloatingActionButton(
          onPressed: _increment,
          tooltip: 'Increment',
          child: const Icon(Icons.add),
        ),
      ),
    );
  }
}

class _HomeTab extends StatelessWidget {
  const _HomeTab({required this.counter});

  final int counter;

  @override
  Widget build(BuildContext context) {
    return Column(
      children: [
        const Padding(
          padding: EdgeInsets.all(16),
          child: Text('Tap increment to increase the counter'),
        ),
        Text(
          '$counter',
          style: Theme.of(context).textTheme.displayMedium,
        ),
        const SizedBox(height: 24),
        Expanded(
          child: ListView.builder(
            itemCount: 5,
            itemBuilder: (context, index) {
              return ListTile(
                title: const Text('Notification'),
                subtitle: Text('Home feed item $index'),
                trailing: IconButton(
                  icon: const Icon(Icons.chevron_right),
                  onPressed: () {
                    Navigator.push(
                      context,
                      MaterialPageRoute(
                        builder: (context) => NotificationDetailPage(index: index),
                      ),
                    );
                  },
                ),
                onTap: () {
                  Navigator.push(
                    context,
                    MaterialPageRoute(
                      builder: (context) => NotificationDetailPage(index: index),
                    ),
                  );
                },
              );
            },
          ),
        ),
      ],
    );
  }
}

class NotificationDetailPage extends StatefulWidget {
  final int index;
  const NotificationDetailPage({super.key, required this.index});

  @override
  State<NotificationDetailPage> createState() => _NotificationDetailPageState();
}

class _NotificationDetailPageState extends State<NotificationDetailPage> {
  final _replyController = TextEditingController();
  bool _isPinned = false;
  String _statusMessage = '';

  @override
  void dispose() {
    _replyController.dispose();
    super.dispose();
  }

  void _submit() {
    if (_replyController.text.isNotEmpty) {
      setState(() {
        _statusMessage = 'Reply submitted: "${_replyController.text}"' + (_isPinned ? ' (Pinned)' : '');
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text('Details'),
      ),
      body: Padding(
        padding: const EdgeInsets.all(24.0),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            Text(
              'Detail view for Home feed item ${widget.index}',
              style: Theme.of(context).textTheme.titleLarge,
            ),
            const SizedBox(height: 16),
            const Text(
              'This is a complex notification that requires user feedback. Please type a reply and decide if you want to pin this notification to your home dashboard.',
            ),
            const SizedBox(height: 24),
            TextField(
              controller: _replyController,
              decoration: const InputDecoration(
                labelText: 'Reply message',
                hintText: 'Enter your response...',
                border: OutlineInputBorder(),
              ),
            ),
            const SizedBox(height: 16),
            SwitchListTile(
              title: const Text('Pin notification'),
              subtitle: const Text('Keep this notification at the top of the feed'),
              value: _isPinned,
              onChanged: (bool value) {
                setState(() {
                  _isPinned = value;
                });
              },
            ),
            const SizedBox(height: 24),
            FilledButton(
              onPressed: _submit,
              child: const Text('Submit'),
            ),
            if (_statusMessage.isNotEmpty) ...[
              const SizedBox(height: 24),
              Container(
                padding: const EdgeInsets.all(16),
                decoration: BoxDecoration(
                  color: Colors.green.shade50,
                  border: Border.all(color: Colors.green),
                  borderRadius: BorderRadius.circular(8),
                ),
                child: Text(
                  _statusMessage,
                  style: const TextStyle(color: Colors.green, fontWeight: FontWeight.bold),
                ),
              ),
            ],
          ],
        ),
      ),
    );
  }
}


class _InboxTab extends StatelessWidget {
  const _InboxTab({
    required this.alertsEnabled,
    required this.searchController,
    required this.onAlertsChanged,
  });

  final bool alertsEnabled;
  final TextEditingController searchController;
  final ValueChanged<bool> onAlertsChanged;

  @override
  Widget build(BuildContext context) {
    return Column(
      children: [
        Padding(
          padding: const EdgeInsets.all(16),
          child: TextField(
            controller: searchController,
            decoration: const InputDecoration(
              hintText: 'Search messages',
              border: OutlineInputBorder(),
            ),
          ),
        ),
        SwitchListTile(
          title: const Text('Enable alerts'),
          value: alertsEnabled,
          onChanged: onAlertsChanged,
        ),
        const Padding(
          padding: EdgeInsets.symmetric(horizontal: 16),
          child: Row(
            children: [
              Chip(label: Text('All')),
              SizedBox(width: 8),
              Chip(label: Text('Unread')),
            ],
          ),
        ),
        Expanded(
          child: ListView.builder(
            itemCount: 12,
            itemBuilder: (context, index) {
              return ListTile(
                leading: const CircleAvatar(child: Icon(Icons.person)),
                title: const Text('Notification'),
                subtitle: Text('Inbox thread $index'),
                trailing: IconButton(
                  icon: const Icon(Icons.delete_outline),
                  onPressed: () {},
                ),
              );
            },
          ),
        ),
      ],
    );
  }
}
